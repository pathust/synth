const path = require("path");

module.exports = {
    apps: [
        {
            name: "fetch-daemon",
            interpreter: path.join(__dirname, ".venv", "bin", "python"),
            script: path.join(__dirname, "synth", "miner", "fetch_daemon.py"),
            // Keep concurrency low to avoid API rate limits.
            args: "--workers 2",
            cwd: __dirname,
            env: {
                PYTHONPATH: ".",
                // DuckDB is optional; disable mirroring by default to avoid hard dependency.
                SYNTH_ENABLE_DUCKDB_WRITER: "0",
            },
            autorestart: true,
            max_restarts: 10,
            restart_delay: 10000,
            log_date_format: "YYYY-MM-DD HH:mm:ss",
        },
    ],
};
