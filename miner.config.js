module.exports = {
  apps: [
    {
      name: "miner",
      interpreter: "python3",
      script: "./neurons/miner.py",
      args: "--netuid 50 --logging.debug --wallet.name quadra_1 --wallet.hotkey l1 --axon.port 60000 --blacklist.force_validator_permit true --blacklist.validator_min_stake 1000",
      env: {
        PYTHONPATH: ".",
      },
    },
  ],
};
