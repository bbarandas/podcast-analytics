from unittest.mock import patch, MagicMock
import pandas as pd
import pytest
from modules.data_sources.statsbomb import list_competitions, list_matches, get_events, get_360


@patch("modules.data_sources.statsbomb.sb.competitions")
def test_list_competitions_returns_dataframe(mock_comps):
    mock_comps.return_value = pd.DataFrame([{"competition_id": 11, "season_id": 1, "competition_name": "La Liga"}])
    result = list_competitions()
    assert isinstance(result, pd.DataFrame)
    assert "competition_id" in result.columns


@patch("modules.data_sources.statsbomb.sb.matches")
def test_list_matches_passes_ids(mock_matches):
    mock_matches.return_value = pd.DataFrame([{"match_id": 3788741}])
    result = list_matches(competition_id=11, season_id=1)
    mock_matches.assert_called_once_with(competition_id=11, season_id=1)
    assert isinstance(result, pd.DataFrame)


@patch("modules.data_sources.statsbomb.sb.events")
def test_get_events_returns_dataframe(mock_events):
    mock_events.return_value = pd.DataFrame([{"type": "Pass", "player": "Messi", "team": "Barcelona"}])
    result = get_events(match_id=3788741)
    mock_events.assert_called_once_with(match_id=3788741)
    assert isinstance(result, pd.DataFrame)


@patch("modules.data_sources.statsbomb.sb.frames")
def test_get_360_returns_dataframe(mock_frames):
    mock_frames.return_value = pd.DataFrame([{"id": "abc", "freeze_frame": []}])
    result = get_360(match_id=3788741)
    mock_frames.assert_called_once_with(match_id=3788741)
    assert isinstance(result, pd.DataFrame)
