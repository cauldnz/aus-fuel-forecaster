# Podman networking fails after switching networks

When the host laptop moves between networks (corporate office, home Wi-Fi, tethered hotspot), containers running under Podman lose DNS resolution even though the host itself has working internet. This document covers the symptom, a fast diagnosis, and the inline fix.

## Symptom

Inside a Podman container, outbound DNS fails:

- `uv sync` hangs or aborts trying to reach `pypi.org`
- `pip install <pkg>` times out
- `curl https://example.com` from inside the container fails with `Could not resolve host`
- `podman exec <container> nslookup pypi.org` fails

Meanwhile, on the host:

- Browser works fine
- `nslookup pypi.org` from PowerShell resolves immediately
- Other non-containerised tools have full network access

## Diagnose

Confirm it's DNS (not routing) by comparing host vs. container resolution:

```powershell
nslookup pypi.org                                    # host: works
podman exec <container> nslookup pypi.org            # container: fails
```

If the container query fails while the host query succeeds, it's the stale-resolver problem documented below.

You can also peek at what the Podman machine thinks its nameservers are:

```powershell
wsl -d podman-machine-default -u root -- cat /etc/resolv.conf
```

If the listed `nameserver` entries are from a previous network (e.g. the office DNS server when you're now at home), that confirms it.

## Fix

Detect the host's current active DNS servers, rewrite the Podman machine's `/etc/resolv.conf`, and restart the machine:

```powershell
# Get host DNS (the active ones, not the cached ones)
$dns = (Get-DnsClientServerAddress -AddressFamily IPv4 |
        Where-Object {$_.ServerAddresses -and $_.InterfaceAlias -notlike "*Loopback*"} |
        Select-Object -ExpandProperty ServerAddresses -First 2) -join " "

# Rewrite WSL distro's resolv.conf (podman-machine-default is the default name)
wsl -d podman-machine-default -u root -- sh -c "echo 'nameserver $($dns -replace ' ', '`nnameserver ')' > /etc/resolv.conf"

# Restart Podman machine to pick up the change
podman machine stop
podman machine start
```

After the restart, re-run the failing container command. DNS should resolve.

This fix is needed **every time** the host network changes. There is no permanent fix short of disabling Podman's DNS caching at the machine level, which has its own trade-offs (loss of internal name resolution between containers, slower lookups). For day-to-day laptop use, the inline rewrite is the pragmatic answer.

## Why it happens

Podman on Windows runs a WSL2 distribution (`podman-machine-default` by default) which hosts the container runtime. When the WSL distro boots, it captures the host's then-current DNS configuration into its own `/etc/resolv.conf`. WSL does not watch the host's DNS state, so when the host hands out new resolvers via DHCP after a network change, the Podman machine keeps using the old ones until it's restarted.

Containers inherit DNS from the Podman machine, so a stale resolver at the machine level becomes a stale resolver in every container — even ones started after the network change. Restarting individual containers does not help; only restarting the Podman machine (after fixing its `resolv.conf`) does.
