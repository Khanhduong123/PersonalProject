from fastapi import APIRouter, HTTPException
from backend.inferences import Inferecences

router = APIRouter(prefix="/user", tags=["user"])


@router.post("/")
async def user_input(sentences: str):
    try:
        inf = Inferecences(
            file_path="data\eng-vie.txt",
            lang1="eng",
            lang2="viet",
            encoder_path="backend\models\encoder.pth",
            decoder_path="backend\models\decoder.pth",
        )
        result = inf.user_input(sentences)

        result = " ".join(result)
    except Exception as e:
        raise HTTPException(status_code=404, detail="Something went wrong")
    return result
