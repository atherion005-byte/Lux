import sys, json
data = json.load(sys.stdin)
big_gpus = [m for m in data if m.get('gpu_ram', 0) >= 48000]
big_gpus.sort(key=lambda x: x.get('dph_total', 999))
print(f'Found {len(big_gpus)} offers with >= 48GB VRAM:')
header = f'{"ID":<12} {"GPU":<20} {"VRAM":<8} {"$/hr":<8} {"vCPUs":<6} {"RAM_GB":<8} {"Location":<25}'
print(header)
for m in big_gpus[:25]:
    gpu = m.get('gpu_name', '?')
    vram = m.get('gpu_ram', 0) / 1000
    price = m.get('dph_total', 0)
    vcpus = m.get('cpu_cores_effective', 0)
    ram = m.get('cpu_ram', 0) / 1000
    loc = m.get('geolocation', '?')
    mid = m.get('id', '?')
    print(f'{mid:<12} {gpu:<20} {vram:.0f}GB    {price:<8.4f} {vcpus:<6.0f} {ram:<8.0f} {loc:<25}')
