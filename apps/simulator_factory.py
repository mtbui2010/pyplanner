"""
simulator_factory.py

Factory class to create AI2-THOR controllers for different environments (iTHOR, ProcTHOR).
"""

import ai2thor.controller
import random
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimulatorFactory:
    @staticmethod
    def create_controller(simulator_type: str = "thor", **kwargs):
        """
        Creates and returns an AI2-THOR Controller based on simulator_type.

        Args:
            simulator_type (str): "thor" (or "ithor") for default scenes,
                                  "procthor" for procedurally generated scenes.
            **kwargs: Other parameters passed to Controller (e.g., width, height, gridSize).

        Returns:
            ai2thor.controller.Controller: Initialized Controller.
        """
        simulator_type = simulator_type.lower().strip()
        
        # General default parameters if not provided
        default_params = {
            "width": 600,
            "height": 600,
            "renderDepthImage": True,
            "renderInstanceSegmentation": True
        }
        # Update kwargs with default params if missing
        for k, v in default_params.items():
            if k not in kwargs:
                kwargs[k] = v

        if simulator_type in ["thor", "ithor"]:
            return SimulatorFactory._create_thor_controller(**kwargs)
        
        elif simulator_type == "procthor":
            return SimulatorFactory._create_procthor_controller(**kwargs)
        
        else:
            raise ValueError(f"Invalid simulator type '{simulator_type}'. Choose 'thor' or 'procthor'.")

    @staticmethod
    def _create_thor_controller(**kwargs):
        """Initializes controller for standard iTHOR environment."""
        logger.info("Initializing iTHOR Controller...")
        
        # If scene not specified, choose default FloorPlan1
        if "scene" not in kwargs:
            kwargs["scene"] = "FloorPlan1"
            
        controller = ai2thor.controller.Controller(**kwargs)
        logger.info(f"iTHOR ready at scene: {kwargs.get('scene')}")
        return controller

    @staticmethod
    def _create_procthor_controller(**kwargs):
        """Initializes controller for ProcTHOR environment."""
        logger.info("Initializing ProcTHOR Controller...")
        
        # ProcTHOR usually requires 'prior' library to load dataset
        try:
            import prior
        except ImportError:
            logger.error("Missing 'prior' library. Please install: pip install prior")
            raise ImportError("Need to install 'prior' library to run ProcTHOR.")

        # Load ProcTHOR-10k dataset (mini or full version)
        # Using 'train' split by default
        dataset = prior.load_dataset("procthor-10k")
        
        # Randomly select a house from train set
        house = dataset["train"][random.randint(0, len(dataset["train"]) - 1)]
        
        # ProcTHOR needs controller initialization first, then reset with scene object
        # Remove 'scene' parameter from kwargs if present, as we use house object
        if "scene" in kwargs:
            del kwargs["scene"]

        # Initialize Controller (default agentMode="default" or can be set to "locobot")
        controller = ai2thor.controller.Controller(**kwargs)
        
        # Reset controller with ProcTHOR scene
        controller.reset(scene=house)
        
        logger.info("ProcTHOR ready with a random house.")
        return controller