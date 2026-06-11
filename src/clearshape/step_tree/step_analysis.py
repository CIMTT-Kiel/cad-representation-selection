from steputils import p21


# %%
# TODO refactor make this a validation method of step tree class
def entity_counts(stepfile: p21.StepFile):
    """Returns dict coutaining counts of unique entities."""
    count_by_entity_name = {}
    # entity_names_unique = unique(stepfile)
    for _, instance in stepfile.data[0].instances.items():
        if isinstance(instance, p21.ComplexEntityInstance):
            for entity in instance.entities:
                if entity.name not in count_by_entity_name.keys():
                    count_by_entity_name[entity.name] = 1
                else:
                    count_by_entity_name[entity.name] += 1

        if isinstance(instance, p21.SimpleEntityInstance):
            if instance.entity.name not in count_by_entity_name.keys():
                count_by_entity_name[instance.entity.name] = 1
            else:
                count_by_entity_name[instance.entity.name] += 1

    return count_by_entity_name
