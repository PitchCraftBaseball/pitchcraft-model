from model_shared.db import find_table_for_column

COLUMN_TABLE_OVERRIDES = {
    "sz_top": "historical_pitches",
    "sz_bot": "historical_pitches",
}


def validate_feature_list_file(filename: str):
    count: int = 0
    table_to_features_map: dict[str, list[str]] = {}
    with open(filename, "r") as file:
        for feature in file:
            if feature.startswith("#"):
                continue
            feature = feature.strip()
            if feature in COLUMN_TABLE_OVERRIDES:
                table_name = COLUMN_TABLE_OVERRIDES[feature]
            else:
                table_name = find_table_for_column("public", feature)

            if table_name is None:
                print(f"Couldn't find a table that contained the feature: {feature}")
                return None
            count += 1
            if table_name in table_to_features_map:
                table_to_features_map[table_name].append(feature)
            else:
                table_to_features_map[table_name] = [feature]
    print(f"<feature_file> successfully validated. Feature count: {count}")
    return table_to_features_map
