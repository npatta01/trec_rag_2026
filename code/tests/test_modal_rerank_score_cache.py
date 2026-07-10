import ast
from pathlib import Path


RUNNER_PATH = Path("code/tools/modal_rerank_score_cache.py")


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _spawn_keywords(function: ast.FunctionDef, remote_name: str) -> dict[str, ast.expr]:
    for node in ast.walk(function):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        owner = node.func.value
        if (
            node.func.attr == "spawn"
            and isinstance(owner, ast.Name)
            and owner.id == remote_name
        ):
            return {
                keyword.arg: keyword.value
                for keyword in node.keywords
                if keyword.arg is not None
            }
    raise AssertionError(f"{function.name} does not spawn {remote_name}")


def test_local_entrypoints_pass_selected_modal_identity_to_remote_status_writers():
    tree = ast.parse(RUNNER_PATH.read_text(encoding="utf-8"))
    score = _function(tree, "score_all_candidates")
    warm = _function(tree, "verify_warm_cache")
    for remote in (score, warm):
        argument_names = {argument.arg for argument in remote.args.args}
        assert "selected_app_name" in argument_names
        assert "selected_volume_name" in argument_names

    main_keywords = _spawn_keywords(_function(tree, "main"), "score_all_candidates")
    verify_keywords = _spawn_keywords(_function(tree, "verify"), "verify_warm_cache")
    for keywords in (main_keywords, verify_keywords):
        assert ast.dump(keywords["selected_app_name"]) == ast.dump(
            ast.Name(id="APP_NAME", ctx=ast.Load())
        )
        assert ast.dump(keywords["selected_volume_name"]) == ast.dump(
            ast.Name(id="VOLUME_NAME", ctx=ast.Load())
        )
