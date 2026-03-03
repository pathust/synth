module.exports = {
    apps: [
        {
            name: "fetch-daemon",
            interpreter: "/Users/taiphan/miniconda3/envs/synth/bin/python",
            script: "./synth/miner/fetch_daemon.py",
            cwd: "/Users/taiphan/Documents/synth",
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
