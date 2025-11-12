# FAQ

## 1. My building time seems long

You can speed up the process by using a lightweight embedding model. Add this to your arguments:

```bash
--embedding-model sentence-transformers/all-MiniLM-L6-v2
```
**Model sizes:** `all-MiniLM-L6-v2` (30M parameters), `facebook/contriever` (~100M parameters), `Qwen3-0.6B` (600M parameters)

## 2. When should I use prompt templates?

**Use prompt templates ONLY with task-specific embedding models** like Google's EmbeddingGemma. These models are specially trained to use different prompts for documents vs queries.

**DO NOT use with regular models** like `nomic-embed-text`, `text-embedding-3-small`, or `bge-base-en-v1.5` - adding prompts to these models will corrupt the embeddings.

**Example usage with EmbeddingGemma:**
```bash
# Build with document prompt
leann build my-docs --embedding-prompt-template "title: none | text: "

# Search with query prompt
leann search my-docs --query "your question" --embedding-prompt-template "task: search result | query: "
```

See the [Configuration Guide: Task-Specific Prompt Templates](configuration-guide.md#task-specific-prompt-templates) for detailed usage.

## 3. Why is LM Studio loading multiple copies of my model?

This was fixed in recent versions. LEANN now properly unloads models after querying metadata, respecting your LM Studio JIT auto-evict settings.

**If you still see duplicates:**
- Update to the latest LEANN version
- Restart LM Studio to clear loaded models
- Check that you have JIT auto-evict enabled in LM Studio settings

**How it works now:**
1. LEANN loads model temporarily to get context length
2. Immediately unloads after query
3. LM Studio JIT loads model on-demand for actual embeddings
4. Auto-evicts per your settings

## 4. Do I need Node.js and @lmstudio/sdk?

**No, it's completely optional.** LEANN works perfectly fine without them using a built-in token limit registry.

**Benefits if you install it:**
- Automatic context length detection for LM Studio models
- No manual registry maintenance
- Always gets accurate token limits from the model itself

**To install (optional):**
```bash
npm install -g @lmstudio/sdk
```

See [Configuration Guide: LM Studio Auto-Detection](configuration-guide.md#lm-studio-auto-detection-optional) for details.
