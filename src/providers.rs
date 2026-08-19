//! Provider capability registry (epic nidus-54l, ticket .9).

/// What a provider can be used for. A provider may support more than one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Capability {
    Embed,
    Summarize,
    Rerank,
}

impl Capability {
    /// Human-facing name, used in error messages ("available embed providers: …").
    pub fn as_str(&self) -> &'static str {
        match self {
            Capability::Embed => "embed",
            Capability::Summarize => "summarize",
            Capability::Rerank => "rerank",
        }
    }
}

/// A provider we ship, tagged with the capabilities it offers.
pub struct ProviderInfo {
    pub name: &'static str,
    pub capabilities: &'static [Capability],
}

/// The canonical list of providers nidus ships.
pub const PROVIDERS: &[ProviderInfo] = &[
    ProviderInfo {
        name: "voyage",
        capabilities: &[Capability::Embed, Capability::Rerank],
    },
    ProviderInfo {
        name: "openai",
        capabilities: &[Capability::Embed, Capability::Summarize],
    },
    ProviderInfo {
        name: "ollama",
        capabilities: &[Capability::Embed],
    },
    ProviderInfo {
        name: "cohere",
        capabilities: &[Capability::Embed, Capability::Rerank],
    },
    ProviderInfo {
        name: "gemini",
        capabilities: &[Capability::Embed],
    },
    ProviderInfo {
        name: "mistral",
        capabilities: &[Capability::Embed],
    },
    ProviderInfo {
        name: "jina",
        capabilities: &[Capability::Embed, Capability::Rerank],
    },
    ProviderInfo {
        name: "openai-compat",
        capabilities: &[Capability::Embed],
    },
    ProviderInfo {
        name: "anthropic",
        capabilities: &[Capability::Summarize],
    },
];

/// Whether `name` is a known provider that offers `cap`.
pub fn supports(name: &str, cap: Capability) -> bool {
    PROVIDERS
        .iter()
        .any(|p| p.name == name && p.capabilities.contains(&cap))
}

/// Whether `name` is a known provider at all (for any capability).
pub fn is_known(name: &str) -> bool {
    PROVIDERS.iter().any(|p| p.name == name)
}

/// The names of every provider offering `cap`, in registry order — used to
/// build helpful "available providers: …" error messages.
pub fn names_with(cap: Capability) -> Vec<&'static str> {
    PROVIDERS
        .iter()
        .filter(|p| p.capabilities.contains(&cap))
        .map(|p| p.name)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn voyage_embeds_and_reranks_but_does_not_summarize() {
        assert!(supports("voyage", Capability::Embed));
        assert!(supports("voyage", Capability::Rerank));
        assert!(!supports("voyage", Capability::Summarize));
    }

    #[test]
    fn anthropic_summarizes_but_does_not_embed() {
        assert!(supports("anthropic", Capability::Summarize));
        assert!(!supports("anthropic", Capability::Embed));
    }

    #[test]
    fn anthropic_does_not_rerank() {
        assert!(!supports("anthropic", Capability::Rerank));
    }

    #[test]
    fn openai_does_both() {
        assert!(supports("openai", Capability::Embed));
        assert!(supports("openai", Capability::Summarize));
    }

    #[test]
    fn openai_does_not_rerank() {
        assert!(!supports("openai", Capability::Rerank));
    }

    #[test]
    fn unknown_provider_supports_nothing() {
        assert!(!supports("does-not-exist", Capability::Embed));
        assert!(!supports("does-not-exist", Capability::Summarize));
        assert!(!supports("does-not-exist", Capability::Rerank));
        assert!(!is_known("does-not-exist"));
    }

    #[test]
    fn names_with_lists_embedders_in_registry_order() {
        assert_eq!(
            names_with(Capability::Embed),
            vec![
                "voyage",
                "openai",
                "ollama",
                "cohere",
                "gemini",
                "mistral",
                "jina",
                "openai-compat"
            ]
        );
    }

    #[test]
    fn names_with_lists_summarizers() {
        assert_eq!(
            names_with(Capability::Summarize),
            vec!["openai", "anthropic"]
        );
    }

    #[test]
    fn names_with_lists_rerankers_in_registry_order() {
        assert_eq!(
            names_with(Capability::Rerank),
            vec!["voyage", "cohere", "jina"]
        );
    }
}
