const path = require("path");

module.exports = {
    apps: [
        {
            name: "fetch-daemon",
            interpreter: path.join(__dirname, ".venv", "bin", "python"),
            script: path.join(__dirname, "synth", "miner", "fetch_daemon.py"),
            args: "--workers 6",
            cwd: __dirname,
            env: {
                PYTHONPATH: ".",
            },
            autorestart: true,
            max_restarts: 10,
            restart_delay: 10000,
            log_date_format: "YYYY-MM-DD HH:mm:ss",
        },
    ],
};
