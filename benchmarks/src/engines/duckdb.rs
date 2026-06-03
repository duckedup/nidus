//! DuckDB adapter — exact brute-force KNN via the built-in `array_cosine_similarity`
//! over a fixed-size `FLOAT[dim]` ARRAY column. No VSS/HNSW extension: a full scan,
//! the apples-to-apples match for nidus.
//!
//! Inserts use Arrow `RecordBatch` + `Appender::append_record_batch` (the row appender
//! can't build array columns). Queries bind the probe as `dim` scalar f32 params and
//! reconstruct it with `array_value(...)` (array *params* aren't supported by the
//! bindings, but scalars are), so there's no per-query SQL rebuild.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use duckdb::Connection;
use duckdb::arrow::array::{ArrayRef, FixedSizeListArray, Float32Array, UInt64Array};
use duckdb::arrow::datatypes::{DataType, Field, Schema};
use duckdb::arrow::record_batch::RecordBatch;

use crate::VectorStore;
use crate::metrics::disk_bytes;

pub struct DuckdbEngine {
    conn: Connection,
    dir: PathBuf,
    dim: usize,
    /// Prepared SELECT with `dim` `?` placeholders for the probe vector.
    search_sql: String,
}

fn vec_field(dim: usize) -> Field {
    Field::new(
        "vec",
        DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, false)), dim as i32),
        false,
    )
}

impl VectorStore for DuckdbEngine {
    const NAME: &'static str = "duckdb";

    fn create(dim: usize, dir: &Path) -> Result<Self> {
        let db_path = dir.join("duck.db");
        let conn = Connection::open(&db_path).context("open duckdb")?;
        conn.execute_batch(&format!(
            "CREATE TABLE items (id UBIGINT, vec FLOAT[{dim}]);"
        ))
        .context("create table")?;

        // `array_value(?, ?, …)` rebuilds the probe from dim scalar params, cast to the
        // column's fixed-size type so `array_cosine_similarity` sees matching shapes.
        let placeholders = vec!["?"; dim].join(",");
        let search_sql = format!(
            "SELECT id, array_cosine_similarity(vec, array_value({placeholders})::FLOAT[{dim}]) AS score \
             FROM items ORDER BY score DESC LIMIT ?;"
        );

        Ok(Self {
            conn,
            dir: dir.to_path_buf(),
            dim,
            search_sql,
        })
    }

    fn ingest(&mut self, ids: &[u64], vectors: &[f32]) -> Result<()> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::UInt64, false),
            vec_field(self.dim),
        ]));

        let id_arr = UInt64Array::from(ids.to_vec());
        let values = Float32Array::from(vectors.to_vec());
        let list = FixedSizeListArray::try_new(
            Arc::new(Field::new("item", DataType::Float32, false)),
            self.dim as i32,
            Arc::new(values),
            None,
        )
        .context("build FixedSizeList")?;

        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(id_arr) as ArrayRef, Arc::new(list) as ArrayRef],
        )
        .context("build record batch")?;

        let mut appender = self.conn.appender("items").context("open appender")?;
        appender
            .append_record_batch(batch)
            .context("append_record_batch")?;
        appender.flush().context("flush appender")?;
        Ok(())
    }

    fn search(&self, query: &[f32], top_k: usize) -> Result<Vec<(u64, f32)>> {
        let mut stmt = self.conn.prepare_cached(&self.search_sql)?;
        // Bind the dim probe components, then top_k for the LIMIT.
        let mut params: Vec<&dyn duckdb::ToSql> = Vec::with_capacity(self.dim + 1);
        for x in query {
            params.push(x as &dyn duckdb::ToSql);
        }
        let k = top_k as i64;
        params.push(&k as &dyn duckdb::ToSql);

        let rows = stmt.query_map(params.as_slice(), |row| {
            let id: u64 = row.get(0)?;
            let score: f32 = row.get(1)?;
            Ok((id, score))
        })?;

        let mut out = Vec::with_capacity(top_k);
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    fn disk_bytes(&self) -> u64 {
        // Flush WAL into the main file so the on-disk size is representative.
        let _ = self.conn.execute_batch("CHECKPOINT;");
        disk_bytes(&self.dir)
    }
}
