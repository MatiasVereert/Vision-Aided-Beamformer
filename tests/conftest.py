import pytest

def pytest_addoption(parser):
    """Añade la opción --plot a pytest."""
    parser.addoption(
        "--plot", action="store_true", default=False, help="Muestra los gráficos de las pruebas"
    )

@pytest.fixture
def plot(request):
    """Fixture para obtener el valor de la opción --plot."""
    return request.config.getoption("--plot")
