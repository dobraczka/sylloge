from nox_poetry import Session, session


@session()
def tests(session: Session) -> None:
    args = session.posargs or ["--cov", "--cov-report=xml"]
    session.install(".")
    session.install("strawman")
    session.install("pytest")
    session.install("pytest-cov")
    session.install("pytest-mock")
    session.run("pytest", *args)


locations = ["sylloge", "tests", "noxfile.py"]


@session(tags=["not test", "style"])  # type: ignore[call-overload]
def lint(session: Session) -> None:
    session.install("pre-commit")
    session.run(
        "pre-commit",
        "run",
        "--all-files",
        "--hook-stage=manual",
        *session.posargs,
    )


@session(tags=["not test", "style"])  # type: ignore[call-overload]
def style_checking(session: Session) -> None:
    args = session.posargs or locations
    session.install("ruff")
    session.run("ruff", "check", *args)


@session()
def pyroma(session: Session) -> None:
    session.install("poetry-core>=1.0.0")
    session.install("pyroma")
    session.run("pyroma", "--min", "10", ".")


@session(tags=["not test", "style"])  # type: ignore[call-overload]
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


@session(tags=["not test"])  # type: ignore[call-overload]
def build_docs(session: Session) -> None:
    session.install(".[docs]")
    session.install("sphinx")
    session.install("insegel")
    session.install("sphinx-automodapi")
    session.install("sphinx-autodoc-typehints")
    session.cd("docs")
    session.run("make", "clean", external=True)
    session.run("make", "html", external=True)
