import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from penaltyblog.viz.dimensions import PitchDimensions
from penaltyblog.viz.pitch import Pitch
from penaltyblog.viz.theme import Theme


@pytest.fixture
def default_pitch():
    """Fixture providing a default Pitch instance with StatsBomb dimensions."""
    return Pitch(provider="statsbomb")


@pytest.fixture
def custom_pitch():
    """Fixture providing a Pitch instance with custom parameters."""
    return Pitch(
        provider="wyscout",
        width=1000,
        height=800,
        theme="dark",
        orientation="vertical",
        view="left",
        title="Test Pitch",
        subtitle="Test Subtitle",
        subnote="Test Subnote",
        show_axis=True,
        show_legend=True,
        show_spots=False,
    )


@pytest.fixture
def sample_dataframe():
    """Fixture providing a sample DataFrame for plotting tests."""
    return pd.DataFrame(
        {
            "x": [50, 60, 70],
            "y": [30, 40, 50],
            "label": ["Point A", "Point B", "Point C"],
        }
    )


class TestPitchInitialization:
    """Tests for Pitch class initialization."""

    def test_default_initialization(self, default_pitch):
        """Test that Pitch initializes with default parameters."""
        assert default_pitch.width == 600
        assert default_pitch.height == 500
        assert default_pitch.orientation == "horizontal"
        assert default_pitch.view == "full"
        assert default_pitch.show_axis is False
        assert default_pitch.show_legend is False
        assert default_pitch.show_spots is True
        assert isinstance(default_pitch.theme, Theme)
        assert isinstance(default_pitch.dim, PitchDimensions)
        assert isinstance(default_pitch.fig, go.Figure)
        assert isinstance(default_pitch.layers, dict)
        assert len(default_pitch.layers) > 0  # Base pitch elements should be added

    def test_custom_initialization(self, custom_pitch):
        """Test that Pitch initializes with custom parameters."""
        assert custom_pitch.width == 1000
        assert custom_pitch.height == 800
        assert custom_pitch.orientation == "vertical"
        assert custom_pitch.view == "left"
        assert custom_pitch.title == "Test Pitch"
        assert custom_pitch.subtitle == "Test Subtitle"
        assert custom_pitch.subnote == "Test Subnote"
        assert custom_pitch.show_axis is True
        assert custom_pitch.show_legend is True
        assert custom_pitch.show_spots is False
        assert custom_pitch.theme.name == "dark"
        assert isinstance(custom_pitch.dim, PitchDimensions)

    def test_custom_dimensions_initialization(self):
        """Test that Pitch initializes with custom PitchDimensions instance."""
        custom_dim = PitchDimensions(length=100, width=50)
        pitch = Pitch(provider=custom_dim)
        assert pitch.dim is custom_dim
        assert pitch.L == custom_dim.get_draw_length()
        assert pitch.W == custom_dim.get_draw_width()

    def test_invalid_view_raises_error(self):
        """Test that invalid view parameter raises ValueError."""
        with pytest.raises(ValueError, match="Unknown view"):
            Pitch(view="invalid_view")


class TestPitchLayerManagement:
    """Tests for layer management methods."""

    def test_add_layer(self, default_pitch, sample_dataframe):
        """Test that _add_layer adds items to the specified layer."""
        # Create a trace to add
        trace = go.Scatter(x=[1, 2, 3], y=[4, 5, 6])

        # Initial layer count
        initial_layers = len(default_pitch.layers)
        initial_traces = len(default_pitch.fig.data)

        # Add the trace to a new layer
        default_pitch._add_layer("test_layer", trace)

        # Check that the layer was added
        assert "test_layer" in default_pitch.layers
        assert len(default_pitch.layers["test_layer"]) == 1
        assert default_pitch.layers["test_layer"][0] is trace

        # Check that the trace was added to the figure
        assert len(default_pitch.fig.data) == initial_traces + 1

    def test_set_layer_visibility(self, default_pitch):
        """Test that set_layer_visibility shows/hides layers."""
        # Create a layer with a trace
        trace = go.Scatter(x=[1, 2, 3], y=[4, 5, 6])
        default_pitch._add_layer("visibility_test", trace)

        # Hide the layer
        default_pitch.set_layer_visibility("visibility_test", visible=False)

        # Check that the trace is not in the figure data
        assert trace not in default_pitch.fig.data

        # Show the layer again
        default_pitch.set_layer_visibility("visibility_test", visible=True)

        # Check that a trace with the same data is in the figure
        # Note: The implementation might not use the same object instance
        assert any(
            isinstance(t, go.Scatter)
            and list(t.x) == [1, 2, 3]
            and list(t.y) == [4, 5, 6]
            for t in default_pitch.fig.data
        )

        # Test with non-existent layer
        with pytest.raises(ValueError, match="Layer 'nonexistent' does not exist"):
            default_pitch.set_layer_visibility("nonexistent", visible=True)

    def test_remove_layer(self, default_pitch):
        """Test that remove_layer removes a layer completely."""
        # Create a layer with a trace
        trace = go.Scatter(x=[1, 2, 3], y=[4, 5, 6])
        default_pitch._add_layer("remove_test", trace)

        # Verify the layer exists
        assert "remove_test" in default_pitch.layers
        assert trace in default_pitch.fig.data

        # Remove the layer
        default_pitch.remove_layer("remove_test")

        # Check that the layer is gone
        assert "remove_test" not in default_pitch.layers
        assert trace not in default_pitch.fig.data

        # Test removing non-existent layer
        with pytest.raises(ValueError, match="Layer 'nonexistent' does not exist"):
            default_pitch.remove_layer("nonexistent")

    def test_set_layer_order(self, default_pitch):
        """Test that set_layer_order reorders layers correctly."""
        # Create multiple layers
        trace1 = go.Scatter(x=[1, 2], y=[3, 4])
        trace2 = go.Scatter(x=[5, 6], y=[7, 8])
        trace3 = go.Scatter(x=[9, 10], y=[11, 12])

        default_pitch._add_layer("layer1", trace1)
        default_pitch._add_layer("layer2", trace2)
        default_pitch._add_layer("layer3", trace3)

        # Reorder the layers
        default_pitch.set_layer_order(["layer3", "layer1", "layer2"])

        # Check the order of layers in the dictionary
        layers_keys = list(default_pitch.layers.keys())
        assert layers_keys.index("layer3") < layers_keys.index("layer1")
        assert layers_keys.index("layer1") < layers_keys.index("layer2")

        # Test with missing layer
        with pytest.raises(ValueError, match="Layers not found"):
            default_pitch.set_layer_order(["layer3", "nonexistent"])


class TestPitchPlottingMethods:
    """Tests for plotting methods."""

    def test_plot_scatter(self, default_pitch, sample_dataframe):
        """Test that plot_scatter returns a Scatter trace and adds it to the 'scatter' layer."""
        # Initial state
        initial_layers = default_pitch.layers.copy()

        # Plot scatter
        result = default_pitch.plot_scatter(
            sample_dataframe, x="x", y="y", hover="label", return_trace=True
        )

        # Check the return value
        assert isinstance(result, go.Scatter)
        assert result.mode == "markers"

        # Check that the layer was updated when return_trace=False
        default_pitch.plot_scatter(sample_dataframe, x="x", y="y")
        assert "scatter" in default_pitch.layers
        assert len(default_pitch.layers["scatter"]) > 0
        assert isinstance(default_pitch.layers["scatter"][0], go.Scatter)

    def test_plot_heatmap(self, default_pitch, sample_dataframe):
        """Test that plot_heatmap returns a Histogram2d trace and adds it to the 'heatmap' layer."""
        # Plot heatmap with return_trace=True
        result = default_pitch.plot_heatmap(
            sample_dataframe, x="x", y="y", bins=(5, 5), return_trace=True
        )

        # Check the return value
        assert isinstance(result, go.Histogram2d)

        # Check that the layer was updated when return_trace=False
        default_pitch.plot_heatmap(sample_dataframe, x="x", y="y")
        assert "heatmap" in default_pitch.layers
        assert len(default_pitch.layers["heatmap"]) > 0
        assert isinstance(default_pitch.layers["heatmap"][0], go.Histogram2d)

    def test_plot_kde(self, default_pitch, sample_dataframe):
        """Test that plot_kde returns a Heatmap trace and adds it to the 'kde' layer."""
        # Plot KDE with return_trace=True
        result = default_pitch.plot_kde(
            sample_dataframe, x="x", y="y", grid_size=50, return_trace=True
        )

        # Check the return value
        assert isinstance(result, go.Heatmap)

        # Check that the layer was updated when return_trace=False
        default_pitch.plot_kde(sample_dataframe, x="x", y="y")
        assert "kde" in default_pitch.layers
        assert len(default_pitch.layers["kde"]) > 0
        assert isinstance(default_pitch.layers["kde"][0], go.Heatmap)

    def test_plot_comets(self, default_pitch):
        """Test that plot_comets returns a list of Scatter traces and adds them to the 'comets' layer."""
        # Create a dataframe with start and end points
        df = pd.DataFrame(
            {
                "x": [10, 20, 30],
                "y": [40, 50, 60],
                "x2": [15, 25, 35],
                "y2": [45, 55, 65],
                "label": ["A", "B", "C"],
            }
        )

        # Plot comets with return_trace=True
        result = default_pitch.plot_comets(
            df,
            x="x",
            y="y",
            x_end="x2",
            y_end="y2",
            hover="label",
            segments=5,
            return_trace=True,
        )

        # Check the return value
        assert isinstance(result, list)
        assert all(isinstance(trace, go.Scatter) for trace in result)
        assert len(result) == 3 * 5  # 3 points × 5 segments

        # Check that the layer was updated when return_trace=False
        default_pitch.plot_comets(df, x="x", y="y", x_end="x2", y_end="y2")
        assert "comets" in default_pitch.layers
        assert len(default_pitch.layers["comets"]) > 0
        assert all(
            isinstance(trace, go.Scatter) for trace in default_pitch.layers["comets"]
        )

    def test_plot_arrows(self, default_pitch):
        """Test that plot_arrows returns a list of annotation dicts and adds them to the 'arrows' layer."""
        # Create a dataframe with start and end points
        df = pd.DataFrame(
            {
                "x": [10, 20, 30],
                "y": [40, 50, 60],
                "x2": [15, 25, 35],
                "y2": [45, 55, 65],
                "label": ["A", "B", "C"],
            }
        )

        # Plot arrows with return_trace=True
        result = default_pitch.plot_arrows(
            df, x="x", y="y", x_end="x2", y_end="y2", hover="label", return_trace=True
        )

        # Check the return value
        assert isinstance(result, list)
        assert all(isinstance(annot, dict) for annot in result)
        assert all("showarrow" in annot for annot in result)
        assert len(result) == 3  # 3 arrows

        # Check that the layer was updated when return_trace=False
        default_pitch.plot_arrows(df, x="x", y="y", x_end="x2", y_end="y2")
        assert "arrows" in default_pitch.layers
        assert len(default_pitch.layers["arrows"]) > 0
        assert all(isinstance(annot, dict) for annot in default_pitch.layers["arrows"])
        assert all("showarrow" in annot for annot in default_pitch.layers["arrows"])


class TestPitchSaveMethod:
    """Tests for the save method of the Pitch class."""

    def test_save_method_exists(self, default_pitch):
        """Test that the save method exists and is callable."""
        assert hasattr(default_pitch, "save")
        assert callable(default_pitch.save)

    @pytest.mark.parametrize(
        "file_format", ["png", "jpg", "jpeg", "webp", "svg", "pdf", "eps"]
    )
    def test_save_format_inference(
        self, default_pitch, file_format, monkeypatch, tmp_path
    ):
        """Test that the save method correctly infers the format from the file extension."""
        # Mock the write_image method to avoid actual file writing
        mock_calls = []

        def mock_write_image(filename, format=None, **kwargs):
            mock_calls.append((filename, format, kwargs))

        monkeypatch.setattr(default_pitch.fig, "write_image", mock_write_image)

        # Call save with a filename that has the specified extension
        test_file = tmp_path / f"test_pitch.{file_format}"
        default_pitch.save(str(test_file))

        # Check that write_image was called with the correct format
        assert len(mock_calls) == 1
        filename, format, kwargs = mock_calls[0]
        assert filename == str(test_file)
        assert format == file_format

    def test_save_explicit_format(self, default_pitch, monkeypatch, tmp_path):
        """Test that the save method uses the explicitly provided format."""
        # Mock the write_image method
        mock_calls = []

        def mock_write_image(filename, format=None, **kwargs):
            mock_calls.append((filename, format, kwargs))

        monkeypatch.setattr(default_pitch.fig, "write_image", mock_write_image)

        # Call save with an explicit format
        test_file = tmp_path / "test_pitch.out"
        default_pitch.save(str(test_file), format="png")

        # Check that write_image was called with the explicit format
        assert len(mock_calls) == 1
        filename, format, kwargs = mock_calls[0]
        assert filename == str(test_file)
        assert format == "png"

    def test_save_invalid_extension(self, default_pitch, tmp_path):
        """Test that the save method raises ValueError for unrecognized file extensions."""
        test_file = tmp_path / "test_pitch.unknown"
        with pytest.raises(ValueError, match="Could not infer format"):
            default_pitch.save(str(test_file))

    def test_save_custom_dimensions(self, default_pitch, monkeypatch):
        """Test that the save method uses custom dimensions when provided."""
        # Mock the write_image method
        mock_calls = []

        def mock_write_image(filename, format=None, **kwargs):
            mock_calls.append((filename, format, kwargs))

        monkeypatch.setattr(default_pitch.fig, "write_image", mock_write_image)

        # Call save with custom dimensions
        custom_width = 1200
        custom_height = 900
        custom_scale = 2.0
        default_pitch.save(
            "test.png", width=custom_width, height=custom_height, scale=custom_scale
        )

        # Check that write_image was called with the custom dimensions
        assert len(mock_calls) == 1
        filename, format, kwargs = mock_calls[0]
        assert kwargs["width"] == custom_width
        assert kwargs["height"] == custom_height
        assert kwargs["scale"] == custom_scale

    def test_save_default_dimensions(self, default_pitch, monkeypatch):
        """Test that the save method uses the pitch's dimensions when not provided."""
        # Mock the write_image method
        mock_calls = []

        def mock_write_image(filename, format=None, **kwargs):
            mock_calls.append((filename, format, kwargs))

        monkeypatch.setattr(default_pitch.fig, "write_image", mock_write_image)

        # Call save without custom dimensions
        default_pitch.save("test.png")

        # Check that write_image was called with the pitch's dimensions
        assert len(mock_calls) == 1
        filename, format, kwargs = mock_calls[0]
        assert kwargs["width"] == default_pitch.width
        assert kwargs["height"] == default_pitch.height
        assert kwargs["scale"] == 1.0

    def test_save_additional_kwargs(self, default_pitch, monkeypatch):
        """Test that the save method passes additional kwargs to write_image."""
        # Mock the write_image method
        mock_calls = []

        def mock_write_image(filename, format=None, **kwargs):
            mock_calls.append((filename, format, kwargs))

        monkeypatch.setattr(default_pitch.fig, "write_image", mock_write_image)

        # Call save with additional kwargs
        default_pitch.save("test.png", engine="kaleido", validate=False)

        # Check that write_image was called with the additional kwargs
        assert len(mock_calls) == 1
        filename, format, kwargs = mock_calls[0]
        assert kwargs["engine"] == "kaleido"
        assert kwargs["validate"] == False


if __name__ == "__main__":
    pytest.main(["-v", "test_pitch.py"])
