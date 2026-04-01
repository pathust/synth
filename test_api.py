from synth.miner.synthdata_client import SynthDataClient
import datetime

client = SynthDataClient()
print("Starting fetch...")
end_dt = datetime.datetime.now(datetime.timezone.utc)
start_dt = end_dt - datetime.timedelta(days=1)
start_str = start_dt.replace(microsecond=0).isoformat().replace('+00:00', 'Z')
end_str = end_dt.replace(microsecond=0).isoformat().replace('+00:00', 'Z')

try:
    data = client.get_historical_validation_scores(
        asset="BTC",
        start_date=start_str,
        end_date=end_str,
        time_length=86400,
        time_increment=300
    )
    print(f"Got {len(data)} items")
    if data:
        print(data[0])
except Exception as e:
    print("Error:", e)
