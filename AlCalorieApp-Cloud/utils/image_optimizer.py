#!/usr/bin/env python3
"""
Image Optimization Utility for YOLO11m Food Detection
Optimizes image size and quality for perfect food item detection
"""

import logging
import numpy as np
from typing import Tuple, Optional
from PIL import Image, ImageEnhance, ImageFilter
import cv2

logger = logging.getLogger(__name__)

class ImageOptimizer:
    """
    Advanced image optimization for YOLO11m food detection
    """
    
    def __init__(self):
        # Optimal dimensions for YOLO11m detection
        self.optimal_width = 1024
        self.optimal_height = 1024
        self.min_size = 512
        self.max_size = 2048
        
        # Quality enhancement parameters
        self.contrast_factor = 1.1
        self.sharpness_factor = 1.05
        self.brightness_factor = 1.02
        self.saturation_factor = 1.03
    
    def optimize_for_detection(self, image: Image.Image, target_size: Optional[Tuple[int, int]] = None) -> Image.Image:
        """
        Optimize image for perfect YOLO11m food detection
        
        Args:
            image: Input PIL Image
            target_size: Optional specific target size (width, height)
            
        Returns:
            Optimized PIL Image
        """
        try:
            original_size = image.size
            logger.info(f"Optimizing image from {original_size}")
            
            # Step 1: Resize to optimal dimensions
            resized_image = self._resize_optimally(image, target_size)
            
            # Step 2: Enhance image quality
            enhanced_image = self._enhance_quality(resized_image)
            
            # Step 3: Apply noise reduction if needed
            cleaned_image = self._reduce_noise(enhanced_image)
            
            logger.info(f"Image optimization complete: {original_size} -> {cleaned_image.size}")
            return cleaned_image
            
        except Exception as e:
            logger.error(f"Image optimization failed: {e}")
            return image
    
    def _resize_optimally(self, image: Image.Image, target_size: Optional[Tuple[int, int]] = None) -> Image.Image:
        """
        Resize image to optimal dimensions while maintaining aspect ratio
        """
        original_width, original_height = image.size
        
        if target_size:
            target_width, target_height = target_size
        else:
            # Calculate optimal size based on aspect ratio
            aspect_ratio = original_width / original_height
            
            if aspect_ratio > 1:  # Landscape
                target_width = min(self.optimal_width, original_width)
                target_height = int(target_width / aspect_ratio)
            else:  # Portrait or square
                target_height = min(self.optimal_height, original_height)
                target_width = int(target_height * aspect_ratio)
        
        # Ensure minimum size for detection
        if target_width < self.min_size:
            target_width = self.min_size
            target_height = int(target_width * (original_height / original_width))
        if target_height < self.min_size:
            target_height = self.min_size
            target_width = int(target_height * (original_width / original_height))
        
        # Ensure maximum size to prevent memory issues
        if target_width > self.max_size:
            target_width = self.max_size
            target_height = int(target_width * (original_height / original_width))
        if target_height > self.max_size:
            target_height = self.max_size
            target_width = int(target_height * (original_width / original_height))
        
        # Resize with high-quality resampling
        resized_image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
        
        logger.info(f"Resized to {target_width}x{target_height}")
        return resized_image
    
    def _enhance_quality(self, image: Image.Image) -> Image.Image:
        """
        Enhance image quality for better detection
        """
        try:
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Enhance contrast for better edge detection
            contrast_enhancer = ImageEnhance.Contrast(image)
            enhanced_image = contrast_enhancer.enhance(self.contrast_factor)
            
            # Enhance sharpness for better detail recognition
            sharpness_enhancer = ImageEnhance.Sharpness(enhanced_image)
            enhanced_image = sharpness_enhancer.enhance(self.sharpness_factor)
            
            # Enhance brightness slightly
            brightness_enhancer = ImageEnhance.Brightness(enhanced_image)
            enhanced_image = brightness_enhancer.enhance(self.brightness_factor)
            
            # Enhance color saturation for better food recognition
            color_enhancer = ImageEnhance.Color(enhanced_image)
            enhanced_image = color_enhancer.enhance(self.saturation_factor)
            
            return enhanced_image
            
        except Exception as e:
            logger.warning(f"Quality enhancement failed: {e}")
            return image
    
    def _reduce_noise(self, image: Image.Image) -> Image.Image:
        """
        Reduce noise while preserving important details
        """
        try:
            # Convert to numpy array for OpenCV processing
            img_array = np.array(image)
            
            # Apply bilateral filter to reduce noise while preserving edges
            denoised = cv2.bilateralFilter(img_array, 9, 75, 75)
            
            # Convert back to PIL Image
            denoised_image = Image.fromarray(denoised)
            
            return denoised_image
            
        except Exception as e:
            logger.warning(f"Noise reduction failed: {e}")
            return image
    
    def create_multiple_scales(self, image: Image.Image, scales: list = [0.75, 1.0, 1.25]) -> list:
        """
        Create multiple scaled versions of the image for comprehensive detection
        
        Args:
            image: Input PIL Image
            scales: List of scale factors
            
        Returns:
            List of (scaled_image, scale_factor) tuples
        """
        scaled_images = []
        
        for scale in scales:
            try:
                # Calculate new dimensions
                new_width = int(image.width * scale)
                new_height = int(image.height * scale)
                
                # Ensure minimum size
                if new_width >= self.min_size and new_height >= self.min_size:
                    scaled_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    scaled_images.append((scaled_image, scale))
                    
            except Exception as e:
                logger.warning(f"Failed to create scale {scale}: {e}")
        
        return scaled_images
    
    def optimize_for_specific_food_types(self, image: Image.Image, food_type: str = "general") -> Image.Image:
        """
        Optimize image based on specific food type for better detection
        
        Args:
            image: Input PIL Image
            food_type: Type of food ("fruits", "vegetables", "meat", "dairy", "general")
            
        Returns:
            Optimized PIL Image
        """
        try:
            # Base optimization
            optimized_image = self.optimize_for_detection(image)
            
            # Apply food-specific enhancements
            if food_type == "fruits":
                # Enhance color saturation for fruits
                color_enhancer = ImageEnhance.Color(optimized_image)
                optimized_image = color_enhancer.enhance(1.2)
                
            elif food_type == "vegetables":
                # Enhance contrast for vegetables
                contrast_enhancer = ImageEnhance.Contrast(optimized_image)
                optimized_image = contrast_enhancer.enhance(1.15)
                
            elif food_type == "meat":
                # Enhance brightness for meat detection
                brightness_enhancer = ImageEnhance.Brightness(optimized_image)
                optimized_image = brightness_enhancer.enhance(1.1)
                
            elif food_type == "dairy":
                # Enhance sharpness for dairy products
                sharpness_enhancer = ImageEnhance.Sharpness(optimized_image)
                optimized_image = sharpness_enhancer.enhance(1.1)
            
            return optimized_image
            
        except Exception as e:
            logger.error(f"Food-specific optimization failed: {e}")
            return image

# Global instance for easy access
image_optimizer = ImageOptimizer()

def optimize_image_for_detection(image: Image.Image, target_size: Optional[Tuple[int, int]] = None) -> Image.Image:
    """
    Convenience function for image optimization
    
    Args:
        image: Input PIL Image
        target_size: Optional specific target size
        
    Returns:
        Optimized PIL Image
    """
    return image_optimizer.optimize_for_detection(image, target_size)

def create_multiple_scales(image: Image.Image, scales: list = [0.75, 1.0, 1.25]) -> list:
    """
    Convenience function for creating multiple scales
    
    Args:
        image: Input PIL Image
        scales: List of scale factors
        
    Returns:
        List of (scaled_image, scale_factor) tuples
    """
    return image_optimizer.create_multiple_scales(image, scales)
