import importlib
import time
import sys
from loguru import logger
from .exceptions import RecoverableException
from .api import FedLedger
from .config import load_config_for_task
from .modules.base import BaseValidationModule, BaseConfig, BaseInputData, BaseMetrics

class ValidationRunner:
    """
    Runs the validation process for a given module and set of task IDs.
    Handles assignment fetching, validation, error handling, and result submission.
    """
    def __init__(
        self,
        module: str,
        task_ids: list[str],
        flock_api_key: str,
        hf_token: str,
        time_sleep: int = 180,
        assignment_lookup_interval: int = 180,
        debug: bool = False,
    ):
        """
        Initialize the ValidationRunner.
        Args:
            module: The name of the validation module to use.
            task_ids: List of task IDs to validate.
            flock_api_key: API key for Flock.
            hf_token: HuggingFace token (passed for compatibility, not used here).
            time_sleep: Time to sleep between retries (seconds).
            assignment_lookup_interval: Assignment lookup interval (seconds).
            debug: Enable debug mode (currently unused).
        """
        self.module = module
        self.task_ids = task_ids
        self.flock_api_key = flock_api_key
        self.hf_token = hf_token
        self.time_sleep = time_sleep
        self.assignment_lookup_interval = assignment_lookup_interval
        self.debug = debug
        self.api = FedLedger(flock_api_key)
        self._setup_modules()

    def _setup_modules(self):
        """Dynamically import and initialize validation modules for each task."""
        all_tasks = self.api.list_tasks()
        tasks = [task for task in all_tasks if task["id"] in self.task_ids]
        task_types = {task["id"]: task["task_type"] for task in tasks}
        if not all(task["task_type"] == self.module for task in tasks):
            raise ValueError(f"Module {self.module} is not valid for the given task ids. Check task types: {task_types}")
        module_mod = importlib.import_module(f"validator.modules.{self.module}")
        module_cls: type[BaseValidationModule] = module_mod.MODULE
        # Map config to module instance, and task_id to module instance
        self.module_config_to_module: dict[BaseConfig, BaseValidationModule] = {}
        self.task_id_to_module: dict[str, BaseValidationModule] = {}
        for task_id in self.task_ids:
            config = load_config_for_task(task_id, self.module, module_cls.config_schema)
            self.module_config_to_module.setdefault(config, module_cls(config=config))
            self.task_id_to_module[task_id] = self.module_config_to_module[config]

    def perform_validation(self, assignment_id: str, task_id: str,input_data: BaseInputData) -> BaseMetrics | None:
        """
        Perform validation for a given assignment and input data.
        """
        module_obj = self.task_id_to_module[task_id]
        for attempt in range(3):
            try:
                return module_obj.validate(input_data)
            except KeyboardInterrupt:
                sys.exit(1)
            except RecoverableException as e:
                logger.error(f"Recoverable exception: {e}")
                sys.exit(1)
            except (RuntimeError, ValueError) as e:
                logger.error(e)
                self.api.mark_assignment_as_failed(assignment_id)
                sys.exit(1)
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {e}")
                if attempt == 2:
                    logger.error(f"Marking assignment {assignment_id} as failed after 3 attempts")
                    self.api.mark_assignment_as_failed(assignment_id)
                    return None

    def run(self):
        """
        Run the validation loop for all configured task IDs.
        This method blocks and runs indefinitely.
        """
        last_successful_request_time = {task_id: time.time() for task_id in self.task_ids}
        while True:
            for task_id in self.task_ids:
                resp = None
                # Try to get a valid assignment
                while True:
                    resp = self.api.request_validation_assignment(task_id)
                    if resp.status_code == 200:
                        last_successful_request_time[task_id] = time.time()
                        break
                    else:
                        # Try to parse JSON response, handle empty or invalid responses
                        try:
                            resp_json = resp.json()
                        except Exception:
                            resp_json = None
                        
                        if resp_json == {"detail": "No task submissions available to validate"}:
                            logger.info("Failed to ask assignment_id: No task submissions available to validate")
                        elif resp_json == {"detail": "Rate limit reached for validation assignment lookup: 1 per 3 minutes"}:
                            time_since_last_success = time.time() - last_successful_request_time[task_id]
                            if time_since_last_success < self.assignment_lookup_interval:
                                time_to_sleep = self.assignment_lookup_interval - time_since_last_success
                                logger.info(f"Sleeping for {int(time_to_sleep)} seconds")
                                time.sleep(time_to_sleep)
                            continue
                        else:
                            logger.error(f"Failed to get assignment: {resp.content}")
                            logger.info(f"Sleeping for {int(self.time_sleep)} seconds")
                            time.sleep(self.time_sleep)
                            continue
                module_obj = self.task_id_to_module[task_id]
                task_submission_data = resp.json()["task_submission"]["data"]
                validation_assignment_data = resp.json()["data"]
                merged_data = {**task_submission_data, **validation_assignment_data}
                input_data = module_obj.input_data_schema.model_validate(merged_data)
                assignment_id = resp.json()["id"]
                metrics = self.perform_validation(assignment_id, task_id, input_data)
                if metrics is None:
                    continue
                resp_submit = self.api.submit_validation_result(
                    assignment_id=assignment_id,
                    data=metrics.model_dump(),
                )
                # if successful, log
                if resp_submit.status_code == 200:
                    logger.info(f"Validation result submitted successfully for assignment {assignment_id}")
                resp_submit.raise_for_status()
