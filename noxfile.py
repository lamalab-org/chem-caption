import nox


# coverage
@nox.session
def coverage(session):
    session.install("coverage")
    session.run("coverage")


# coverage-clean
@nox.session
def coverage_clean(session):
    session.install("coverage")
    session.run("coverage", "erase")


# lint
@nox.session
def lint(session):
    session.install("black[jupyter]")
    session.install("isort")
    session.install("nbqa")
    session.run("black", ".")
    session.run("isort", ".")
    session.run("nbqa", "isort", ".")


# manifest
@nox.session
def manifest(session):
    session.install("check-manifest")
    session.run("check-manifest")


# flake8
@nox.session
def flake8(session):
    session.install("flake8")
    session.run("flake8", "src/")
    session.run("flake8", "tests/")


# mypy
@nox.session
def mypy(session):
    session.install("mypy")
    session.run("mypy", "--install-types", "--non-interactive", "--ignore-missing-imports", "src/")


# docs
@nox.session
def docs(session):
    session.install("-r", "requirements.txt")
    session.install("-r", "test_requirements.txt")
    session.run(
        "python",
        "-m",
        "sphinx",
        "-b",
        "html",
        "-d",
        "docs/build/doctrees",
        "docs/source",
        "docs/build/html",
    )


# doc8
@nox.session
def doc8(session):
    session.install("doc8")
    session.run("doc8", "docs/source/")


# docstr-coverage
@nox.session
def docstr(session):
    session.install("docstr-coverage")
    session.run("docstr-coverage", "src/", "tests/", "--skip-private", "--skip-magic")


# docs-test
@nox.session
def doctest(session):
    session.install("-r", "requirements.txt")
    session.install("-r", "test_requirements.txt")
    session.install("coverage")

    session.run("mkdir", "-p", "tmp", external=True)
    session.run("cp", "-r", "docs/source", "tmp/source", external=True)
    session.run(
        "python",
        "-m",
        "sphinx",
        "-W",
        "-b",
        "html",
        "-d",
        "tmp/build/doctrees",
        "tmp/source",
        "tmp/build/html",
    )
    session.run(
        "python",
        "-m",
        "sphinx",
        "-W",
        "-b",
        "coverage",
        "-d" "tmp/build/doctrees",
        "tmp/source",
        "tmp/build/coverage",
    )
    session.run("cat", "tmp/build/coverage/c.txt", external=True)
    session.run("cat", "tmp/build/coverage/python.txt", external=True)


# py
@nox.session(venv_backend="conda", python=["3.12"])
def test(session):
    session.install("setuptools<80")
    session.install("-r", "requirements.txt")
    session.install("-r", "test_requirements.txt")
    session.install("coverage")
    session.install("git+https://github.com/kjappelbaum/givemeconformer.git")
    session.conda_install("xtb-python", channel="conda-forge")
    session.conda_install("libblas=*=*mkl", channel="conda-forge")
    session.conda_install("qcengine", channel="conda-forge")
    session.conda_install("spyrmsd", channel="conda-forge")
    session.install("geometric")
    session.install("pyberny")

    session.install(".", "--no-deps")

    session.run(
        "coverage", "run", "-p", "-m", "pytest", "--durations=20", "--ignore=tests/regression/"
    )


# coverage-report
@nox.session
def report(session):
    session.install("coverage")
    session.run("coverage", "combine")
    session.run("coverage", "report")
