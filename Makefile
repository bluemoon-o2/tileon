all:
	python3 -m pip install --upgrade pip
	python3 -m pip install -e ".[all]"

dev:
	python3 -m pip install --upgrade pip
	python3 -m pip install -e ".[dev]"

build:
	python3 -m pip install --verbose -e .

build.sdist:
	python3 -m build --sdist --verbose

deploy.pypi:
	python3 -m twine upload dist/*

deploy.gh-docs:
	mkdocs build
	mkdocs gh-deploy

test:
	python3 -m pytest --full-trace -v

clean:
	- rm -rf _skbuild
	- rm lib/*.so
	- rm lib/*.dll
	- rm lib/*.lib

.PHONY: \
	all \
	dev \
	build \
	build.sdist \
	deploy.pypi \
	deploy.gh-docs \
	test \
	clean
