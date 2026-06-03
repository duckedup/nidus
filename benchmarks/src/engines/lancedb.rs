//! LanceDB adapter — exact brute-force KNN via `bypass_vector_index` (no IVF/HNSW),
//! the apples-to-apples match for nidus.
//!
//! LanceDB's API is async; the harness is sync, so this adapter owns a tokio runtime and
//! `block_on`s each call. Inserts go through an Arrow `RecordBatch` (UInt64 ids +
//! FixedSizeList<Float32>); the query stream is collected with `futures`.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
use arrow_array::{
    Array, ArrayRef, FixedSizeListArray, Float32Array, RecordBatch, RecordBatchIterator,
    RecordBatchReader, UInt64Array,
};
use arrow_schema::{DataType, Field, Schema};
use futures::TryStreamExt;
use lancedb::query::{ExecutableQuery, QueryBase};
use lancedb::{Connection, DistanceType, Table, connect};
use tokio::runtime::Runtime;

use crate::VectorStore;
use crate::metrics::disk_bytes;

const TABLE: &str = "items";

pub struct LancedbEngine {
    rt: Runtime,
    _conn: Connection,
    table: Table,
    schema: Arc<Schema>,
    dir: PathBuf,
}

fn schema_for(dim: usize) -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::UInt64, false),
        Field::new(
            "vec",
            DataType::FixedSizeList(
                // Lance's vector columns use a nullable inner "item" field.
                Arc::new(Field::new("item", DataType::Float32, true)),
                dim as i32,
            ),
            false,
        ),
    ]))
}

impl VectorStore for LancedbEngine {
    const NAME: &'static str = "lancedb";

    fn create(dim: usize, dir: &Path) -> Result<Self> {
        let rt = Runtime::new().context("tokio runtime")?;
        let uri = dir.to_string_lossy().to_string();
        let schema = schema_for(dim);

        let (conn, table) = rt.block_on(async {
            let conn = connect(&uri).execute().await.context("connect")?;
            let table = conn
                .create_empty_table(TABLE, schema.clone())
                .execute()
                .await
                .context("create_empty_table")?;
            Ok::<_, anyhow::Error>((conn, table))
        })?;

        Ok(Self {
            rt,
            _conn: conn,
            table,
            schema,
            dir: dir.to_path_buf(),
        })
    }

    fn ingest(&mut self, ids: &[u64], vectors: &[f32]) -> Result<()> {
        let dim = self.schema.field(1).data_type();
        let len = match dim {
            DataType::FixedSizeList(_, n) => *n,
            _ => unreachable!("vec column is a fixed-size list"),
        };

        let id_arr = UInt64Array::from(ids.to_vec());
        let values = Float32Array::from(vectors.to_vec());
        let list = FixedSizeListArray::try_new(
            Arc::new(Field::new("item", DataType::Float32, true)),
            len,
            Arc::new(values),
            None,
        )
        .context("build FixedSizeList")?;

        let batch = RecordBatch::try_new(
            self.schema.clone(),
            vec![Arc::new(id_arr) as ArrayRef, Arc::new(list) as ArrayRef],
        )
        .context("build record batch")?;

        // `Table::add` requires `Scannable`, implemented for the boxed trait object —
        // coerce explicitly rather than passing the concrete iterator type.
        let reader: Box<dyn RecordBatchReader + Send> = Box::new(RecordBatchIterator::new(
            vec![Ok(batch)],
            self.schema.clone(),
        ));
        self.rt
            .block_on(async { self.table.add(reader).execute().await.context("add") })?;
        Ok(())
    }

    fn search(&self, query: &[f32], top_k: usize) -> Result<Vec<(u64, f32)>> {
        let batches: Vec<RecordBatch> = self.rt.block_on(async {
            let stream = self
                .table
                .query()
                .nearest_to(query)
                .context("nearest_to")?
                .distance_type(DistanceType::Cosine)
                .bypass_vector_index()
                .limit(top_k)
                .execute()
                .await
                .context("execute query")?;
            stream.try_collect::<Vec<_>>().await.context("collect")
        })?;

        let mut out = Vec::with_capacity(top_k);
        for batch in &batches {
            let ids = batch
                .column_by_name("id")
                .and_then(|c| c.as_any().downcast_ref::<UInt64Array>())
                .ok_or_else(|| anyhow!("missing/!u64 id column"))?;
            let dist = batch
                .column_by_name("_distance")
                .and_then(|c| c.as_any().downcast_ref::<Float32Array>())
                .ok_or_else(|| anyhow!("missing/!f32 _distance column"))?;
            for i in 0..batch.num_rows() {
                // Cosine distance -> similarity, to match nidus's score scale.
                out.push((ids.value(i), 1.0 - dist.value(i)));
            }
        }
        Ok(out)
    }

    fn disk_bytes(&self) -> u64 {
        disk_bytes(&self.dir)
    }
}
