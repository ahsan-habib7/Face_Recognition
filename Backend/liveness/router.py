from fastapi import APIRouter, UploadFile, File, Query
import numpy as np
import cv2

from .session import create_session, get_session, delete_session
from .service import process_frame_logic

router = APIRouter()

@router.post("/start")
def start():
    sid = create_session()
    return {"session_id": sid, "blink_needed": 3}


@router.post("/frame")
async def frame(
    session_id: str = Query(..., description="Session ID from /liveness/start"),
    frame: UploadFile = File(..., description="JPEG frame from webcam")
):

    session = get_session(session_id)

    if not session:
        return {"error": "Session expired", "stage": "error", "progress": 0,
                "icon": "⚠️", "message": "Session expired — please restart liveness check"}

    img_bytes = await frame.read()
    np_arr    = np.frombuffer(img_bytes, np.uint8)
    image     = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if image is None:
        return {"error": "Could not decode frame", "stage": "error",
                "progress": 0, "icon": "⚠️", "message": "Invalid frame received"}

    return process_frame_logic(session, image)


@router.post("/cancel")
def cancel(session_id: str = Query(..., description="Session ID to cancel")):
    """
    Cancel a liveness session.
    Called when the user closes the camera modal.
    """
    delete_session(session_id)
    return {"cancelled": True}
