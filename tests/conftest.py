import logging

import pytest

from tests.mockdata import generate_fake_rastercube

logger = logging.getLogger(__name__)


@pytest.fixture
def mock_rastercube_factory():
    return generate_fake_rastercube
