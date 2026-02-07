from fastapi import APIRouter, HTTPException
import logging
from src.api.schemas import (
    OTPRequest, VerifyOTPRequest, UserProfile
)
from src.api.dependencies import get_db_manager

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/auth/send-otp")
async def send_otp(request: OTPRequest):
    """
    Send OTP to phone number. 
    MOCK: Returns success and logs OTP.
    """
    try:
        # In real app: Generate random 6 digit code and SMS it.
        # For Demo/Dev: Fixed OTP for specific test numbers or random for others.
        otp_code = "123456" 
        logger.info(f"OTP for {request.phone}: {otp_code}")
        
        return {
            "success": True,
            "message": "OTP sent successfully",
            "debug_otp": otp_code  # Remove in production
        }
    except Exception as e:
        logger.error(f"Send OTP error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/auth/verify-otp")
async def verify_otp(request: VerifyOTPRequest):
    """
    Verify OTP and Login/Register user.
    SPECIAL: Creates/Returns 'Rachit' dummy user for phone 1234567890
    """
    try:
        if request.otp != "123456":
            raise HTTPException(status_code=400, detail="Invalid OTP")
            
        db = get_db_manager()
        
        # Check if user exists
        user = db.get_user_by_phone(request.phone)
        
        # SPECIAL DEMO LOGIC
        if request.phone == "1234567890" and not user:
             # Create dummy user Rachit
            import random
            import sqlite3
            random_acres = round(random.uniform(2.0, 15.0), 1)
            
            # Use raw SQL or create a specialized create method in future
            # For now, leveraging create_user but with dummy email/pass since we haven't refactored create_user fully
            # Actually, let's just insert strictly for this demo or use create_user with placeholders
            
            db.create_user(
                email=f"rachit_{request.phone}@example.com",
                password="dummy_password", # unused in OTP flow
                name="Rachit",
                phone=request.phone
            )
            
            # Update more fields manually if needed (Location: Pune, Acres)
            # Since create_user is basic, we might need a direct update or ensure create_user supports it.
            # For now, let's rely on Profile Page editing or DB defaults, OR update DB manager.
            # But the user asked for dummy data: Location Pune, Acres Random.
            # Let's do a quick update query or extended create.
            
            conn = sqlite3.connect(db.db_path)
            c = conn.cursor()
            c.execute('''
                UPDATE users 
                SET location_name = ?, land_area = ?
                WHERE phone = ?
            ''', ("Pune", random_acres, request.phone))
            conn.commit()
            conn.close()
            
            # Fetch again
            user = db.get_user_by_phone(request.phone)

        # Standard Registration (if not Rachit logic and user doesn't exist)
        if not user:
            # Just create a basic user
            temp_name = request.name or "New Farmer"
            db.create_user(
                email=f"user_{request.phone}@farm.ai",
                password="otp_user_pass",
                name=temp_name,
                phone=request.phone
            )
            user = db.get_user_by_phone(request.phone)
            
        return {
            "success": True,
            "user": user,
            "token": "simulated-otp-jwt-token"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Verify OTP error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/user/profile")
async def save_user_profile(profile: UserProfile):
    """Save or update farmer profile"""
    try:
        db = get_db_manager()
        success = db.save_user_profile(profile.dict())
        if success:
            return {"success": True, "message": "Profile saved", "variety_config": "Loaded specific care for Phule Purandar" if profile.custard_apple_variety == "Phule Purandar" else "Standard care loaded"}
        else:
            raise HTTPException(status_code=500, detail="Failed to save profile")
    except Exception as e:
        logger.error(f"Profile save error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/user/profile/{user_id}")
async def get_user_profile_endpoint(user_id: str):
    """Get farmer profile"""
    try:
        db = get_db_manager()
        profile = db.get_user_profile(user_id)
        if profile:
            return {"success": True, "profile": profile}
        else:
            return {"success": False, "message": "Profile not found"}
    except Exception as e:
        logger.error(f"Profile fetch error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
