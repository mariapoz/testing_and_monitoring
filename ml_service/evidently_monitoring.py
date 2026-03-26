import asyncio
from threading import Lock

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset
from evidently.ui.workspace import RemoteWorkspace


EVIDENTLY_URL = "http://158.160.2.37:8000/"
EVIDENTLY_PROJECT_ID = "019d2af3-3896-72c2-9440-81567ccdd246"


class DriftMonitor:
    def __init__(self) -> None:
        self.lock = Lock()
        self.reference_data = pd.DataFrame()
        self.current_data = pd.DataFrame()

    def set_reference_data(self, df: pd.DataFrame) -> None:
        with self.lock:
            self.reference_data = df.copy()

    def add_record(self, df: pd.DataFrame, prediction: int, probability: float) -> None:
        row = df.copy()
        row["prediction"] = prediction
        row["probability"] = probability

        with self.lock:
            self.current_data = pd.concat([self.current_data, row], ignore_index=True)

    def snapshot(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        with self.lock:
            return self.reference_data.copy(), self.current_data.copy()

    def clear_current(self) -> None:
        with self.lock:
            self.current_data = pd.DataFrame()


DRIFT_MONITOR = DriftMonitor()


async def evidently_worker(period_seconds: int = 60, min_records: int = 10) -> None:
    while True:
        await asyncio.sleep(period_seconds)

        reference_data, current_data = DRIFT_MONITOR.snapshot()

        if reference_data.empty:
            print("Evidently skipped: reference_data is empty")
            continue

        if len(current_data) < min_records:
            print(
                f"Evidently skipped: not enough current records "
                f"({len(current_data)}/{min_records})"
            )
            continue

        try:
            report = Report(metrics=[DataDriftPreset()])
            result = report.run(
                reference_data=reference_data,
                current_data=current_data,
            )

            workspace = RemoteWorkspace(EVIDENTLY_URL)
            workspace.add_run(EVIDENTLY_PROJECT_ID, result)

            print(
                f"Evidently report sent successfully. "
                f"reference={len(reference_data)}, current={len(current_data)}"
            )

            DRIFT_MONITOR.clear_current()

        except Exception as e:
            print(f"Evidently report failed: {e}")