from nox_poetry import Session, session


@session()
def tests(session: Session) -> None:
    args = session.posargs or ["--cov", "--cov-report=xml"]
    session.run_always("poetry", "install", external=True)
    session.install("pytest")
    session.install("pytest-cov")
    session.run("pytest", *args)


locations = ["sylloge", "tests", "noxfile.py"]


@session(tags=["not test", "style"])  # type: ignore
def lint(session: Session) -> None:
    args = session.posargs or locations
    session.install("black", "isort")
    session.run("black", *args)
    session.run("isort", *args)


@session(tags=["not test", "style"])  # type: ignore
def style_checking(session: Session) -> None:
    args = session.posargs or locations
    session.install(
        "pyproject-flake8",
        "flake8-eradicate",
        "flake8-isort",
        "flake8-debugger",
        "flake8-comprehensions",
        "flake8-print",
        "flake8-black",
        "flake8-black",
        "darglint",
        "pydocstyle",
    )
    session.run("pflake8", "--docstring-style", "sphinx", *args)


@session(tags=["not test", "style"])  # type: ignore
def type_checking(session: Session) -> None:
    args = session.posargs or locations
    session.run_always("poetry", "install", external=True)
    session.run(
        "mypy",
        "--install-types",
        "--ignore-missing-imports",
        "--non-interactive",
        *args,
    )


@session(tags=["not test"])  # type: ignore
def build_docs(session: Session) -> None:
    session.install(".[docs]")
    session.install("sphinx")
    session.install("insegel")
    session.install("sphinx-automodapi")
    session.install("sphinx-autodoc-typehints")
    session.cd("docs")
    session.run("make", "clean", external=True)
    session.run("make", "html", external=True)
