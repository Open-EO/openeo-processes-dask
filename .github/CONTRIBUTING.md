# Welcome to openeo-processes-dask

Thank you for your efforts in contributing to our project! Any contribution you make, e.g. bug reports, bug fixes, additional documentation, enhancement suggestions, and other ideas are welcome and will be reflected on [openeo-processes-dask](https://github.com/Open-EO/openeo-processes-dask).

If there are any questions, do not hesitate to ask, by opening an issue or directly contacting us. We also have a biweekly online meeting, where PRs and issues are discussed. Feel free to join! Contact: support@eodc.eu

Our aim is to work on this project together - everyone is welcome to contribute. Please follow our [Code of Conduct](./CODE_OF_CONDUCT.md).

On this page, we will guide you through the contribution workflow from opening an issue and creating a PR to reviewing and merging the PR.


### Getting started

To get an overview of the project, read the [README](../README.md) file. The process implementations are based on the openEO [specification](https://processes.openeo.org/). The aim of this project is, to offer implementations for all listed processes based on the [xarray](https://github.com/pydata/xarray)/[dask](https://github.com/dask/dask) ecosystem.

To get a general introduction to openEO, see:
- [openEO docs](https://docs.openeo.cloud/)
- [openEO API](https://api.openeo.org/)
- [openEO registration](https://docs.openeo.cloud/join/free_trial.html#connect-with-egi-check-in)
- [openEO official website](https://openeo.cloud/)


### Issues and bugs

#### Create a new issue

Reporting bugs is an important part of improving the project. If you find any unclear documentation, unexpected behaviour in the implementation, missing features, etc. first [check if an issue already exists](https://github.com/Open-EO/openeo-processes-dask/issues). If a related issue doesn't exist, you can open a new one.

#### Create a bug report

A bug report should always contain python code, to recreate the behaviour. This can be formatted nicely using ` ```python ... ``` `. Add an explaination on which parts are unexpected. If the issue is related to a certain process, also have a look at the process specification, to check, what kind of results should be produced, which parameters are required, which error messages should be raised, etc.

#### Solve an issue

Scan through our [existing issues](https://github.com/Open-EO/openeo-processes-dask/issues) to find one that interests you. We normally do not assign issues to anyone. If you find an interesting issue to work on, you are welcome to open a PR with a fix.

### Make Changes

#### Version control, git and github

To make changes to the code, you will need a free [github](https://github.com/) account. The code is available on github, where we use [git](https://git-scm.com/) for version control.

#### Make changes locally

1. Create an account and log in to [github](https://github.com/).
2. Fork the repository. Go to the [project](https://github.com/Open-EO/openeo-processes-dask) and click the `Fork` button on the top of the page.
3. Clone the fork of the repository to your local machine.
```
git clone https://github.com/<YOUR USER NAME>/openeo-processes-dask.git
cd openeo-processes-dask
```
4. Set up the development environment using the instructions in the [README](../README.md).
```
poetry install --all-extras
```
5. Create a new branch
```
git checkout -b new-branch-name
```

Once this is set up, you can start making code changes.

### Commit your update

You can check, which files contain changes using `git status`.

Before you commit your changes, make sure your tests run through and update the tests if required. It is recommended to cover all your changes in the tests. Once you submit your changes, github will automatically check, if the new lines of code are covered in the tests.

If you made complex changes, it is helpful to also include comments next to your code, in order to document the changes for reviewers and other contributors.

We are using pre-commit hooks, to stick to a nice structure and formatting. See [pre-commit](https://pre-commit.com/) and [git hooks](https://git-scm.com/book/en/v2/Customizing-Git-Git-Hooks).

Once your changes are ready, you can commit and push them to github.
1. To commit all modified files into the local copy of your repo, do `git commit -am 'A commit message'`.
2. To push the changes up to your forked repo on GitHub, do a `git push`.

### Pull Request

Once you’re ready or need feedback on your code, open a Pull Request, also known as a PR, on the github project page.
- Don't forget to [link PR to issue](https://github.com/Open-EO/openeo-processes-dask/issues) if you are solving one.
- Once you submit your PR, an openeo-processes-dask team member will review your changes. We may ask questions or request additional information.
- We may ask for changes to be made before a PR can be merged.
- As you update your PR and apply changes, mark each conversation as [resolved](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/commenting-on-a-pull-request#resolving-conversations).
- If you run into any merge issues, checkout this [git tutorial](https://github.com/skills/resolve-merge-conflicts) to help you resolve merge conflicts and other issues.

You can directly comment in the PR if you want the reviewer to pay particular attention to something - If your request is not ready to be merged, just say so in your pull request message and create a “Draft PR". This way, you can get preliminary code review and others can see what is currently being worked on.

## New releases

Once the reviewer(s) approve the PR and there are no more changes requested, the PR will be merged into the main branch. A new release will then be created based on the main branch.

There will be a new release at least every two weeks. (In case, there were no new changes at all, the release might be skipped.)

If important changes are added - such as bug fixes and additional processes - new releases might be made in between.

Small changes - such as new comments, updated documentation - will be included in the bi-weekly releases.

## Adding new processes

If you only want to update implementation details in a process, the specification should remain as it is and you do not need to update the submodule.

If you want to add a new process or update the parameters of the process, you will also need to interact with the submodule.

The specifications come from a fork of the official openeo-processes: https://github.com/eodcgmbh/openeo-processes

To add a new process:
- add the specification to https://github.com/eodcgmbh/openeo-processes
    - create a github fork
    - check if the process you want to add is in the missing-processes folder and if so, move it to the root folder
    - if not, create a new process definition
    - create a PR and merge it
- update the submodule in openeo-processes-dask
    - create a github fork of this repository
    - Use `git submodule init` and `git submodule update` in your forked repository to update the specifications
    - To specify the submodule explicitely, you can use
     `git submodule update --remote openeo_processes_dask/specs/openeo-processes/`
    - find more details on submodules [here](https://git-scm.com/book/en/v2/Git-Tools-Submodules)
- add the implementation in openeo-processes-dask
- cover the new implementation in the tests
- update the dependencies, if you need to introduce a new package. `poetry add ...`.
- create a PR to merge your fork into the openeo-processes-dask

New implementations can be tested using the local [client-side-processing](https://open-eo.github.io/openeo-python-client/cookbook/localprocessing.html). This allows testing process without a connection to an openEO back-end on a user's local netCDFs, geoTIFFs, ZARR files, or remote STAC Collections/ Items. 

For backend development, the specifications and implementations can be used to create a process registry, e.g. https://github.com/Open-EO/openeo-pg-parser-networkx/blob/main/examples/01_minibackend_demo.ipynb
```
from openeo_processes_dask.specs import load_collection as load_collection_spec
process_registry["load_collection"] = Process(spec=load_collection_spec, implementation=load_collection)
```

## Prior to submitting a PR - a checklist

- Add comments and documentation for your code
- Make sure your tests still run through and add additional tests.
- Format your code nicely - run `poetry run pre-commit install` and `pre-commit run --all-files`.
- Add a descriptive comment to your commit and push your code to [github](https://github.com/Open-EO/openeo-processes-dask).
- Create a PR with a descriptive title for your changes.
