import os

try:
    from hypothesis import HealthCheck, settings

    settings.register_profile("default", max_examples=60, deadline=None,
                              suppress_health_check=[HealthCheck.too_slow])
    settings.register_profile("stress", max_examples=3000, deadline=None,
                              suppress_health_check=[HealthCheck.too_slow])
    settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "default"))
except ImportError:  # hypothesis is optional
    pass
