module.exports = {
    apps: [
        {
            name: "fetch-daemon",
            interpreter: "/home/user/synth/.venv/bin/python",
            script: "./synth/miner/fetch_daemon.py",
            cwd: "/home/user/synth",
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
