import sys
import click
from validator.validation_runner import ValidationRunner

@click.command()
@click.argument(
    "module",
    type=str,
    required=True,
)
@click.option(
    "--task_ids",
    type=str,
    required=False,
    help="The ids of the task, separated by comma (not required for local test)",
)
@click.option('--flock-api-key', envvar='FLOCK_API_KEY', required=False, help='Flock API key (not required for local test)')
@click.option('--hf-token', envvar='HF_TOKEN', required=False, help='HuggingFace token')
@click.option('--time-sleep', envvar='TIME_SLEEP', default=60 * 3, type=int, show_default=True, help='Time to sleep between retries (seconds)')
@click.option('--assignment-lookup-interval', envvar='ASSIGNMENT_LOOKUP_INTERVAL', default=60 * 3, type=int, show_default=True, help='Assignment lookup interval (seconds)')
@click.option("--debug", is_flag=True)
# Local test options
@click.option('--local-test', is_flag=True, help='Run a local validation test without submitting to FedLedger')
@click.option('--hf-repo', type=str, help='HuggingFace repository ID for local test (e.g., username/model-name)')
@click.option('--revision', type=str, default='main', help='Git revision/commit ID for local test (default: main)')
@click.option('--validation-set', type=str, help='Path to local validation set file for local test')
@click.option('--model-filename', type=str, default='model.onnx', help='Model filename in repo for local test (default: model.onnx)')
@click.option('--max-params', type=int, default=100000000, help='Maximum model parameters for local test (default: 100M)')
def main(
    module: str,
    task_ids: str,
    flock_api_key: str,
    hf_token: str,
    time_sleep: int,
    assignment_lookup_interval: int,
    debug: bool,
    local_test: bool,
    hf_repo: str,
    revision: str,
    validation_set: str,
    model_filename: str,
    max_params: int,
):
    """
    CLI entrypoint for running the validation process.
    Delegates core logic to ValidationRunner.
    
    For local testing, use --local-test flag with --hf-repo and --validation-set options.
    Example: python environment_entrypoint.py rl --local-test --hf-repo username/model --validation-set data.npz
    """
    if local_test:
        # Local test mode - no API required
        if not hf_repo:
            click.echo("Error: --hf-repo is required for local test mode", err=True)
            sys.exit(1)
        
        runner = ValidationRunner(
            module=module,
            local_test=True,
            debug=debug,
        )
        try:
            runner.run_local_test(
                hf_repo=hf_repo,
                revision=revision,
                validation_set_path=validation_set,
                model_filename=model_filename,
                max_params=max_params,
            )
        except KeyboardInterrupt:
            click.echo("\nLocal test interrupted by user.")
            sys.exit(0)
        except Exception as e:
            click.echo(f"\nLocal test failed: {e}", err=True)
            sys.exit(1)
    else:
        # Normal validation mode - requires API
        if not task_ids:
            click.echo("Error: --task_ids is required for normal validation mode", err=True)
            sys.exit(1)
        if not flock_api_key:
            click.echo("Error: --flock-api-key is required for normal validation mode", err=True)
            sys.exit(1)
            
        runner = ValidationRunner(
            module=module,
            task_ids=task_ids.split(","),
            flock_api_key=flock_api_key,
            hf_token=hf_token,
            time_sleep=time_sleep,
            assignment_lookup_interval=assignment_lookup_interval,
            debug=debug,
        )
        try:
            runner.run()
        except KeyboardInterrupt:
            click.echo("\nValidation interrupted by user.")
            sys.exit(0)

if __name__ == "__main__":
    main()
