# sql_from_nl (genericness demo)

A sketch showing the harness core doesn't know about documents. To actually run this example you'd add:

- `inputs/<id>.json` files each containing `{"question": "...", "schema_ddl": "..."}`
- `ground_truth.jsonl` with `{"id": "...", "sql": "SELECT ..."}` per line
- `seed_process.py` that prompts the OpenAI-compat chat endpoint for SQL and returns `{"id": ..., "prediction": {"sql": "..."}}`
- `eval.py` that executes both predicted and expected SQL against a SQLite fixture and compares result sets

The same `autoptim run task.yaml` command works — only the task bundle changes.
