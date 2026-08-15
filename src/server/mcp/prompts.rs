//! MCP prompts: the recall-then-answer pattern shipped with the server, so a client does
//! not reinvent it. The recall runs here, server-side, so the messages come back filled in.

use rmcp::{
    ErrorData as McpError,
    model::{GetPromptResult, Prompt, PromptArgument, PromptMessage, Role},
};
use serde_json::{Map, Value as JsonValue};

// Imported for its methods on `AnyEmbedder` — a trait method, not inherent.
use crate::SearchOpts;
use crate::embed::Embedder;

use super::HitDto;
use super::NidusMcp;
use super::args::{api_error, required_str};
use super::search::with_ttl_guard;

pub(super) const PROMPTS_TTL_MS: u64 = 3_600_000;

/// `top_k` for a prompt, which — unlike a tool — has no schema to tell a client the
/// argument is a number (`PromptArgument` carries only name/description/required), so a
/// compliant client filling a template will often send `"3"`. Accept both.
fn prompt_top_k(args: &Map<String, JsonValue>) -> Result<usize, McpError> {
    let raw = match args.get("top_k") {
        None | Some(JsonValue::Null) => return Ok(crate::server::dto::default_top_k()),
        Some(JsonValue::Number(n)) => n.as_u64(),
        Some(JsonValue::String(s)) => s.trim().parse::<u64>().ok(),
        Some(_) => None,
    };
    let k = raw
        .filter(|n| *n > 0)
        .ok_or_else(|| McpError::invalid_params("`top_k` must be a positive integer", None))?
        as usize;
    if k > crate::server::dto::MAX_TOP_K {
        return Err(McpError::invalid_params(
            format!("`top_k` must not exceed {}", crate::server::dto::MAX_TOP_K),
            None,
        ));
    }
    Ok(k)
}

pub(super) fn prompts() -> Vec<Prompt> {
    vec![Prompt::new(
        "recall_then_answer",
        Some(
            "Recall the memories most relevant to a question and return them already \
             assembled with an instruction to answer from them, citing the ids used, and \
             to say so plainly when memory does not cover the question.",
        ),
        Some(vec![
            PromptArgument::new("question")
                .with_description("The question to answer using what is in memory.")
                .with_required(true),
            PromptArgument::new("collection")
                .with_description("Which collection to recall from. `list_collections` if unsure.")
                .with_required(true),
            PromptArgument::new("top_k")
                .with_description(
                    "How many memories to pull in, as a positive integer. Defaults to the \
                     server's default.",
                )
                .with_required(false),
        ]),
    )]
}

impl NidusMcp {
    pub(super) async fn get_prompt(
        &self,
        args: &Map<String, JsonValue>,
    ) -> Result<GetPromptResult, McpError> {
        let embedder = self.embedder()?;
        let question = required_str(args, "question")?;
        let collection = required_str(args, "collection")?;
        let top_k = prompt_top_k(args)?;

        let vector = embedder
            .embed_query(&question)
            .await
            .map_err(|e| McpError::internal_error(e.to_string(), None))?;

        let opts = SearchOpts {
            top_k,
            filter: with_ttl_guard(None),
            ..Default::default()
        };
        let collection_for_run = collection.clone();
        let hits = crate::server::run_read(self.state.clone(), move |db| {
            crate::memory::guard_recall_identity(db, embedder.as_ref(), &collection_for_run)?;
            db.search(collection_for_run.as_str(), &vector, &opts)
        })
        .await
        .map_err(api_error)?;

        let hits: Vec<HitDto> = hits.into_iter().map(HitDto::from).collect();
        let count = hits.len();
        let text = if hits.is_empty() {
            format!(
                "No memories in `{collection}` matched the question below. Answer it from \
                 your own knowledge, but say plainly that memory held nothing relevant.\n\n\
                 Question: {question}"
            )
        } else {
            let rendered = serde_json::to_string_pretty(&hits).unwrap_or_else(|_| "[]".to_string());
            format!(
                "Memories from `{collection}` relevant to the question:\n\n{rendered}\n\n\
                 Question: {question}\n\n\
                 Answer using only these memories, citing the id(s) you relied on. If they do \
                 not cover the question, say so plainly rather than guessing."
            )
        };

        Ok(
            GetPromptResult::new(vec![PromptMessage::new_text(Role::User, text)]).with_description(
                format!(
                    "Recalled {count} memor{} from `{collection}`.",
                    if count == 1 { "y" } else { "ies" }
                ),
            ),
        )
    }
}
