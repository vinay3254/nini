use std::collections::HashMap;

pub struct Tokenizer {
    chars: Vec<char>,
    stoi: HashMap<char, u32>,
}

impl Tokenizer {
    pub fn from_corpus(text: &str) -> Self {
        let mut chars: Vec<char> = text.chars().collect::<std::collections::HashSet<_>>().into_iter().collect();
        chars.sort();
        let stoi = chars.iter().enumerate().map(|(i, &c)| (c, i as u32)).collect();
        Self { chars, stoi }
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        text.chars().filter_map(|c| self.stoi.get(&c).copied()).collect()
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        ids.iter().map(|&i| self.chars[i as usize]).collect()
    }

    pub fn vocab_size(&self) -> usize {
        self.chars.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_text() {
        let tok = Tokenizer::from_corpus("hello world");
        let ids = tok.encode("hello world");
        assert_eq!(tok.decode(&ids), "hello world");
    }

    #[test]
    fn vocab_size_matches_distinct_chars() {
        let tok = Tokenizer::from_corpus("aabbcc");
        assert_eq!(tok.vocab_size(), 3);
    }

    #[test]
    fn encode_unknown_char_is_skipped() {
        let tok = Tokenizer::from_corpus("abc");
        let ids = tok.encode("abz");
        assert_eq!(ids.len(), 2); // 'z' not in vocab, dropped
    }
}
