import hashlib
from pathlib import Path

PYPI_SHA256_BY_WHEEL = {
    "mediapipe-1.2.2-py3-none-win_amd64.whl": "71f44832fdd32b6270bc4379bcdd934287fb9eab974c3e74ef3fa76c7e79b54a",
    "opencv_contrib_python-5.0.0.93-cp37-abi3-win_amd64.whl": "461622db95c964652d4d8fda171034961c3de270f78a6095aaad31050771774a",
    "matplotlib-3.11.2-cp313-cp313-win_amd64.whl": "bc067c462a86f0e57bf52fc6e058d90171a5420007dac00c46e90153050c69a7",
    "contourpy-1.4.0-cp313-cp313-win_amd64.whl": "6507a016976a75a15aee47809a34605de6244ccc056e6164bf9fb14b27eecad4",
    "cycler-0.12.1-py3-none-any.whl": "85cef7cff222d8644161529808465972e51340599459b8ac3ccbac5a854e0d30",
    "fonttools-4.68.0-cp313-cp313-win_amd64.whl": "6ed796b220654b4d37ad20af73419ec7969b669dd678cab88afb602c5683a331",
    "kiwisolver-1.5.1-cp313-cp313-win_amd64.whl": "dc23390afe9f4ef9ac3bcc72a03a56eebbde03f4c571a32cb38f859cff9a6524",
    "pillow-13.0.0-cp313-cp313-win_amd64.whl": "9dda0788a9932776a75d8bcc020076edc308ce2508b51f6739a45e3158b3fc42",
    "pyparsing-3.3.3-py3-none-any.whl": "ece8c00a69cf01b45d0b1dedabb469c90d8caf996d4fda40f147627a122849a4",
    "python_dateutil-4.0.0-py3-none-any.whl": "a5e246f79eb80475e691bf93862cead98decd8a498cf53ba0516fcdc0f50dc1c",
    "tzdata-2026.6-py2.py3-none-any.whl": "0a10eec6e13ef16b627fabed9a5f2ba61a62b1c67f327305f4e7e412a70b4aec",
}

INSTALLED_PACKAGES = {
    "PIL",
    "contourpy",
    "cv2",
    "cycler",
    "dateutil",
    "fontTools",
    "kiwisolver",
    "matplotlib",
    "mediapipe",
    "mpl_toolkits",
    "pylab.py",
    "pyparsing",
    "tzdata",
}


class WheelError(Exception):
    pass


def verify_unchanged_wheel(wheel_path: Path) -> None:
    expected_sha256 = PYPI_SHA256_BY_WHEEL.get(wheel_path.name)
    if expected_sha256 is None:
        raise WheelError(f"{wheel_path.name} is not in the list of bundled wheels (Tools/bundled_wheels.py).")
    actual_sha256 = hashlib.sha256(wheel_path.read_bytes()).hexdigest()
    if actual_sha256 != expected_sha256:
        raise WheelError(
            f"{wheel_path} differs from the file published on PyPI (SHA-256 {actual_sha256}, expected "
            f"{expected_sha256}). Bundled wheels must be unchanged."
        )
