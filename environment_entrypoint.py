import sys
import json
import click
from validator.validation_runner import ValidationRunner

LOCAL_VALIDATION_MODULES = ("video_inconsistency",)


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
    help="The ids of the task, separated by comma",
)
@click.option('--flock-api-key', envvar='FLOCK_API_KEY', required=False, help='Flock API key')
@click.option('--hf-token', envvar='HF_TOKEN', required=False, help='HuggingFace token')
@click.option('--time-sleep', envvar='TIME_SLEEP', default=60 * 3, type=int, show_default=True, help='Time to sleep between retries (seconds)')
@click.option('--assignment-lookup-interval', envvar='ASSIGNMENT_LOOKUP_INTERVAL', default=60 * 3, type=int, show_default=True, help='Assignment lookup interval (seconds)')
@click.option("--debug", is_flag=True)
@click.option("--local-validation", is_flag=True, help=f"Run one local validation and exit. Supported for: {', '.join(LOCAL_VALIDATION_MODULES)}.")
@click.option("--hg-repo-id", "--hf-model-repo", "local_hg_repo_id", type=str, help="HuggingFace repo id or local model directory for local validation.")
@click.option("--revision", default="main", show_default=True, help="HuggingFace revision for local validation.")
@click.option("--validation-data-url", "--validation-zip-url", "--validation-set-url", "validation_data_url", type=str, help="Validation package path/URL for local validation.")
@click.option("--adapter-filename", type=str, default=None, help="Adapter file inside the model repo (defaults to the module's standard name).")
@click.option("--max-clips", type=int, help="Limit local video validation to the first N clips.")
@click.option("--device", type=str, help="Device for local validation, e.g. cuda or cpu.")
@click.option("--torch-dtype", type=str, help="Torch dtype for local validation.")
@click.option("--output-json", type=str, help="Write local validation metrics JSON to this path.")
def main(
    module: str,
    task_ids: str | None,
    flock_api_key: str | None,
    hf_token: str | None,
    time_sleep: int,
    assignment_lookup_interval: int,
    debug: bool,
    local_validation: bool,
    local_hg_repo_id: str | None,
    revision: str,
    validation_data_url: str | None,
    adapter_filename: str | None,
    max_clips: int | None,
    device: str | None,
    torch_dtype: str | None,
    output_json: str | None,
):
    """
    CLI entrypoint for running the validation process.
    Delegates core logic to ValidationRunner.
    """
    if local_validation:
        if module not in LOCAL_VALIDATION_MODULES:
            raise click.ClickException(
                f"--local-validation is supported only for: {', '.join(LOCAL_VALIDATION_MODULES)}"
            )
        if not local_hg_repo_id:
            raise click.ClickException("--local-validation requires --hg-repo-id/--hf-model-repo")
        if not validation_data_url:
            raise click.ClickException("--local-validation requires --validation-data-url")

        from validator.modules.video_inconsistency.detector import DEFAULT_ADAPTER_FILENAME
        from validator.modules.video_inconsistency.local_validate import run_local_validation

        result = run_local_validation(
            hg_repo_id=local_hg_repo_id,
            revision=revision,
            validation_data_url=validation_data_url,
            adapter_filename=adapter_filename or DEFAULT_ADAPTER_FILENAME,
            max_clips=max_clips,
            device=device,
            torch_dtype=torch_dtype,
            output_json=output_json,
            hf_token=hf_token,
        )
        click.echo(json.dumps(result, indent=2))
        return

    if not task_ids:
        raise click.ClickException("--task_ids is required unless --local-validation is used")
    if not flock_api_key:
        raise click.ClickException("--flock-api-key or FLOCK_API_KEY is required unless --local-validation is used")
    if not hf_token:
        raise click.ClickException("--hf-token or HF_TOKEN is required unless --local-validation is used")

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
