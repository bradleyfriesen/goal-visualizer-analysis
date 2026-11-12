from src import paths

print("Repository:", paths.REPO_ROOT)
print("Workspace:", paths.WORKSPACE_ROOT)
print("Data:", paths.DATA_DIR)
print("Raw:", paths.RAW_DATA_DIR)
print("Processed:", paths.PROCESSED_DATA_DIR)
print("NHL-tracking-pipeline data:", paths.TRACKING_DATA_DIR)
print("Team colours:", paths.COLOUR_PATH)

def test_data_directories_exist():
    assert paths.REPO_ROOT.exists()
    assert paths.WORKSPACE_ROOT.exists()
    assert paths.DATA_DIR.exists()
    assert paths.RAW_DATA_DIR.exists()
    assert paths.PROCESSED_DATA_DIR.exists()
    assert paths.TRACKING_DATA_DIR.exists()
    assert paths.COLOUR_PATH.exists()
