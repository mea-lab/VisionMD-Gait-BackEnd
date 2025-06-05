from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.core.files.storage import FileSystemStorage
from django.conf import settings
import os, time, cv2, json, base64
from datetime import datetime, timezone

@api_view(['POST'])
def update_video_data(request):

    # Get all variables set up and check if folder and file paths exist
    video_id = request.GET.get('id', None)
    if not video_id:
        return Response("Video project id not provided.", status=400)
    
    file_name = request.GET.get('file_name', None)
    if not video_id:
        return Response("File name to hold data not provided.", status=400)
    if not file_name.endswith('.json'):
        return Response("Provided file is not a JSON file.", status=400)

    folder_path = os.path.join(settings.MEDIA_ROOT, "video_uploads")
    project_folder_path = os.path.join(folder_path, video_id)
    if not os.path.isdir(project_folder_path):
        return Response("Video project folder does not exist.", status=400)
    


    try:
        metadata = request.data
        metadata_wrapped = {(file_name.removesuffix('.json')) : metadata}
        if not isinstance(metadata_wrapped, dict):
            raise ValueError("Invalid JSON format. Expected a JSON object.")
    except Exception as e:
        return Response(f"Invalid request body: {str(e)}", status=400)

    # Save metadata.json into the same folder
    data_path = os.path.join(project_folder_path, file_name)
    try:
        with open(data_path, 'w') as jf:
            json.dump(metadata_wrapped, jf, indent=4)
    except Exception as e:
        return Response(f"Error writing to file: {str(e)}", status=500)

    # Return success
    return Response(status=200)
