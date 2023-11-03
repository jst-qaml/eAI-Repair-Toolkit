from pathlib import Path

import repair.dataset
import repair.methods
import repair.model
import repair.utils

# extend napespace path
FIXTURE_PATH = Path(__file__).parent / "fixtures"

repair.dataset.__path__.append(str(FIXTURE_PATH / "dataset"))
repair.methods.__path__.append(str(FIXTURE_PATH / "methods"))
repair.model.__path__.append(str(FIXTURE_PATH / "model"))
repair.utils.__path__.append(str(FIXTURE_PATH / "utils"))
