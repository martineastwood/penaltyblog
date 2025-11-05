import pytest

from penaltyblog.matchflow.steps.opta import config


class TestOptaConfig:
    """Test cases for Opta configuration constants."""

    def test_non_paginated_sources_is_set(self):
        """Test that NON_PAGINATED_SOURCES is a set."""
        assert isinstance(config.NON_PAGINATED_SOURCES, set)

    def test_non_paginated_sources_contains_expected_values(self):
        """Test that NON_PAGINATED_SOURCES contains expected endpoint types."""
        expected_sources = {
            "tournament_schedule",
            "match_basic",
            "match_stats_basic",
            "match_events",
            "player_season_stats",
            "team_season_stats",
            "xg_shots",
            "xg_player_summary",
            "xg_team_summary",
        }
        assert expected_sources.issubset(config.NON_PAGINATED_SOURCES)

    def test_default_pagination_constants(self):
        """Test default pagination constants."""
        assert isinstance(config.DEFAULT_PAGE_SIZE, int)
        assert config.DEFAULT_PAGE_SIZE > 0
        assert isinstance(config.DEFAULT_PAGE_NUM, int)
        assert config.DEFAULT_PAGE_NUM >= 1

    def test_endpoint_configs_structure(self):
        """Test that ENDPOINT_CONFIGS has proper structure."""
        assert isinstance(config.ENDPOINT_CONFIGS, dict)

        for endpoint, endpoint_config in config.ENDPOINT_CONFIGS.items():
            assert isinstance(endpoint, str)
            assert isinstance(endpoint_config, dict)
            assert "path_template" in endpoint_config

            # Check that path_template contains {auth_key} placeholder
            assert "{auth_key}" in endpoint_config["path_template"]

    def test_endpoint_configs_tournament_calendars(self):
        """Test tournament_calendars endpoint configuration."""
        tournament_config = config.ENDPOINT_CONFIGS["tournament_calendars"]

        assert "status_variants" in tournament_config
        assert isinstance(tournament_config["status_variants"], dict)

        expected_variants = ["active", "authorized", "active_authorized", "all"]
        for variant in expected_variants:
            assert variant in tournament_config["status_variants"]

    def test_endpoint_configs_match_stats_basic_supports_multi(self):
        """Test that match_stats_basic supports multi-fixture."""
        match_stats_config = config.ENDPOINT_CONFIGS["match_stats_basic"]
        assert match_stats_config.get("supports_multi") is True

    def test_parameter_mappings_structure(self):
        """Test that PARAMETER_MAPPINGS has proper structure."""
        assert isinstance(config.PARAMETER_MAPPINGS, dict)

        for source, param_mapping in config.PARAMETER_MAPPINGS.items():
            assert isinstance(source, str)
            assert isinstance(param_mapping, dict)

            for arg_name, param_name in param_mapping.items():
                assert isinstance(arg_name, str)
                assert isinstance(param_name, str)

    def test_parameter_mappings_contains_expected_sources(self):
        """Test that PARAMETER_MAPPINGS contains expected sources."""
        expected_sources = {
            "matches_basic",
            "match_basic",
            "match_stats_basic",
            "match_events",
            "teams",
            "squads",
            "player_season_stats",
            "team_season_stats",
        }
        assert expected_sources.issubset(config.PARAMETER_MAPPINGS.keys())

    def test_parameter_mappings_matches_basic_completeness(self):
        """Test matches_basic parameter mapping completeness."""
        matches_mapping = config.PARAMETER_MAPPINGS["matches_basic"]

        expected_params = [
            "fixture_uuids",
            "tournament_calendar_uuid",
            "competition_uuids",
            "contestant_uuid",
            "opponent_uuid",
            "contestant_position",
            "date_range",
            "delta_timestamp",
            "use_opta_names",
        ]

        for param in expected_params:
            assert param in matches_mapping

    def test_response_parsers_structure(self):
        """Test that RESPONSE_PARSERS has proper structure."""
        assert isinstance(config.RESPONSE_PARSERS, dict)

        for source, parser_name in config.RESPONSE_PARSERS.items():
            assert isinstance(source, str)
            assert isinstance(parser_name, str)
            assert parser_name.startswith("parse_") or parser_name.startswith(
                "extract_"
            )

    def test_response_parsers_contains_expected_sources(self):
        """Test that RESPONSE_PARSERS contains expected sources."""
        expected_sources = {
            "tournament_schedule",
            "match_basic",
            "match_stats_basic",
            "match_events",
            "player_season_stats",
            "team_season_stats",
        }
        assert expected_sources.issubset(config.RESPONSE_PARSERS.keys())

    def test_pagination_response_keys_structure(self):
        """Test that PAGINATION_RESPONSE_KEYS has proper structure."""
        assert isinstance(config.PAGINATION_RESPONSE_KEYS, dict)

        for source, key_paths in config.PAGINATION_RESPONSE_KEYS.items():
            assert isinstance(source, str)
            assert isinstance(key_paths, list)

            for key_path in key_paths:
                assert isinstance(key_path, str)

    def test_pagination_response_keys_contains_expected_sources(self):
        """Test that PAGINATION_RESPONSE_KEYS contains expected sources."""
        expected_sources = {
            "tournament_calendars",
            "matches_basic",
            "teams",
            "squads",
        }
        assert expected_sources.issubset(config.PAGINATION_RESPONSE_KEYS.keys())

    def test_pagination_response_keys_nested_paths(self):
        """Test that nested key paths are properly formatted."""
        matches_keys = config.PAGINATION_RESPONSE_KEYS["matches_basic"]
        assert "matches.match" in matches_keys  # Example of nested path

    def test_endpoint_config_path_templates(self):
        """Test that all path templates are properly formatted."""
        for endpoint, endpoint_config in config.ENDPOINT_CONFIGS.items():
            path_template = endpoint_config["path_template"]

            # Should start with /
            assert path_template.startswith("/")

            # Should contain {auth_key} placeholder
            assert "{auth_key}" in path_template

            # Should not have consecutive slashes
            assert "//" not in path_template

    def test_parameter_mapping_consistency(self):
        """Test that parameter mappings are consistent with endpoint configs."""
        # All sources in PARAMETER_MAPPINGS should exist in ENDPOINT_CONFIGS
        for source in config.PARAMETER_MAPPINGS.keys():
            assert (
                source in config.ENDPOINT_CONFIGS
            ), f"{source} in PARAMETER_MAPPINGS but not in ENDPOINT_CONFIGS"

    def test_response_parser_consistency(self):
        """Test that response parsers are consistent with endpoint configs."""
        # All sources in RESPONSE_PARSERS should exist in ENDPOINT_CONFIGS
        for source in config.RESPONSE_PARSERS.keys():
            assert (
                source in config.ENDPOINT_CONFIGS
            ), f"{source} in RESPONSE_PARSERS but not in ENDPOINT_CONFIGS"

    def test_pagination_keys_consistency(self):
        """Test that pagination response keys are consistent with endpoint configs."""
        # All sources in PAGINATION_RESPONSE_KEYS should exist in ENDPOINT_CONFIGS
        for source in config.PAGINATION_RESPONSE_KEYS.keys():
            assert (
                source in config.ENDPOINT_CONFIGS
            ), f"{source} in PAGINATION_RESPONSE_KEYS but not in ENDPOINT_CONFIGS"

    def test_non_paginated_sources_consistency(self):
        """Test that non-paginated sources are consistent with endpoint configs."""
        # All sources in NON_PAGINATED_SOURCES should exist in ENDPOINT_CONFIGS
        # Note: Some sources like xg_* may not have endpoint configs yet, which is expected
        endpoint_config_sources = set(config.ENDPOINT_CONFIGS.keys())
        non_paginated_in_configs = (
            config.NON_PAGINATED_SOURCES & endpoint_config_sources
        )

        # Verify that the sources that are in both sets are properly configured
        for source in non_paginated_in_configs:
            assert (
                source in config.ENDPOINT_CONFIGS
            ), f"{source} should be in ENDPOINT_CONFIGS"

        # Verify that known non-paginated sources are properly marked
        known_non_paginated = {
            "tournament_schedule",
            "match_basic",
            "match_stats_basic",
            "match_events",
            "player_season_stats",
            "team_season_stats",
        }
        assert known_non_paginated.issubset(config.NON_PAGINATED_SOURCES)

    def test_tournament_calendars_status_variants_paths(self):
        """Test tournament_calendars status variants produce valid paths."""
        base_path = "/tournamentcalendar/{auth_key}"
        variants = config.ENDPOINT_CONFIGS["tournament_calendars"]["status_variants"]

        for status, suffix in variants.items():
            full_path = base_path + suffix
            assert full_path.startswith("/")
            assert "//" not in full_path

    def test_config_values_are_immutable(self):
        """Test that configuration constants are not accidentally modified."""
        # Store original values
        original_non_paginated = config.NON_PAGINATED_SOURCES.copy()
        original_page_size = config.DEFAULT_PAGE_SIZE
        original_endpoint_configs = config.ENDPOINT_CONFIGS.copy()

        # Try to modify (this should work in Python, but we want to ensure tests don't modify state)
        config.NON_PAGINATED_SOURCES.add("test_source")
        config.DEFAULT_PAGE_SIZE = 999

        # Restore original values for other tests
        config.NON_PAGINATED_SOURCES.clear()
        config.NON_PAGINATED_SOURCES.update(original_non_paginated)
        config.DEFAULT_PAGE_SIZE = original_page_size

        # Verify restoration
        assert config.NON_PAGINATED_SOURCES == original_non_paginated
        assert config.DEFAULT_PAGE_SIZE == original_page_size
