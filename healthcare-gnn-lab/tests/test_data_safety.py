import numpy as np
import pytest
from healthcare_gnn.data.common import PatientSplit, assert_events_before_index, patient_split
from healthcare_gnn.data.adapters import choose_mimic_demo_target

def test_patient_split_has_no_overlap():
    ids=np.repeat(np.arange(30),2); split=patient_split(ids,7); split.validate(ids)

def test_overlap_is_rejected():
    ids=np.array([1,1,2]); split=PatientSplit(np.array([0]),np.array([1]),np.array([2]))
    with pytest.raises(ValueError,match="Patient leakage"): split.validate(ids)

def test_future_events_are_rejected():
    with pytest.raises(ValueError,match="Temporal leakage"): assert_events_before_index(np.array([4,7]),np.array([5,7]))

def test_demo_target_fallback_is_explicit():
    assert choose_mimic_demo_target(2)=="early_mortality_tutorial"
    assert choose_mimic_demo_target(20)=="next_visit_heart_failure"

import pandas as pd
from healthcare_gnn.data.ehr import history_before_index, is_heart_failure_code

def test_ehr_history_is_strictly_pre_index():
    events=pd.DataFrame({"subject_id":[1,1],"event_time":pd.to_datetime(["2020-01-01","2020-02-01"])})
    result=history_before_index(events,pd.Series({1:pd.Timestamp("2020-01-15")}))
    assert len(result)==1 and is_heart_failure_code("I50.9",10)
