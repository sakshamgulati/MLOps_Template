from evidently.ui.workspace.cloud import CloudWorkspace
import os

ws = CloudWorkspace(
token=os.getenv('EVI_API'),
url="https://app.evidently.cloud")
project = ws.create_project("github actions",team_id="74fdd884-9679-4693-819b-a6695e652e25")
print("Project created successfully")