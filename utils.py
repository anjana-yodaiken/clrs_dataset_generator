import clrs
from clrs._src import specs


def _iterate_sampler(sampler, batch_size):
    while True:
        yield sampler.next(batch_size)


def restore_model(
    processor_type,
    use_ln,
    num_layers,
    nb_heads,
    node_hid_size,
    edge_hid_size_1,
    edge_hid_size_2,
    graph_vec,
    disable_edge_updates,
    save_emb_sub_dir,
    save_embeddings,
    hint_mode,
    head_size,
    ptr_from_edges,
    use_lstm,
    learning_rate,
    checkpoint_path,
    freeze_processor,
    dropout_prob,
    hint_teacher_forcing_noise,
    train_spec,
    val_sampler,
    test_sampler,
    model_seed,
    eval_batch_size,
    encoder_decoder_path,
    model_path,
):
    processor_factory = clrs.get_processor_factory(
        processor_type,
        use_ln=use_ln,
        nb_layers=num_layers,
        nb_heads=nb_heads,
        node_hid_size=node_hid_size,
        edge_hid_size_1=edge_hid_size_1,
        edge_hid_size_2=edge_hid_size_2,
        graph_vec=graph_vec,
        disable_edge_updates=disable_edge_updates,
        save_emb_sub_dir=save_emb_sub_dir,
        save_embeddings=save_embeddings,
    )

    if hint_mode == "encoded_decoded_nodiff":
        encode_hints = True
        decode_hints = True
        decode_diffs = False
    elif hint_mode == "decoded_only_nodiff":
        encode_hints = False
        decode_hints = True
        decode_diffs = False
    elif hint_mode == "encoded_decoded":
        encode_hints = True
        decode_hints = True
        decode_diffs = True
    elif hint_mode == "decoded_only":
        encode_hints = False
        decode_hints = True
        decode_diffs = True
    elif hint_mode == "none":
        encode_hints = False
        decode_hints = False
        decode_diffs = False
    else:
        raise ValueError("Hint mode not in {encoded_decoded, decoded_only, none}.")
    hidden_size = nb_heads * head_size
    model_params = dict(
        processor_factory=processor_factory,
        hidden_dim=192,
        encode_hints=encode_hints,
        decode_hints=decode_hints,
        decode_diffs=decode_diffs,
        ptr_from_edges=ptr_from_edges,
        use_lstm=use_lstm,
        learning_rate=learning_rate,
        checkpoint_path=checkpoint_path,
        freeze_processor=freeze_processor,
        dropout_prob=dropout_prob,
        hint_teacher_forcing_noise=hint_teacher_forcing_noise,
    )

    rt_model = clrs.models.BaselineModel(
        spec=train_spec, dummy_trajectory=val_sampler.next(), **model_params
    )

    feedback = test_sampler.next(eval_batch_size, eval=True)

    rt_model.init(
        feedback.features,
        model_seed + 1,
    )
    rt_model.restore_model(
        str(model_path),
        only_load_processor=False,
        encoder_decoder_path=encoder_decoder_path,
    )
    test_sampler.reset_proc_samples()
    val_sampler.reset_proc_samples()

    return rt_model


def get_dataset_samplers(
    algorithm, train_seed, valid_seed, test_seed, train_size, test_size, valid_size
):
    train_sampler_for_eval = None
    train_sampler, spec = clrs.build_sampler(
        name=algorithm,
        seed=train_seed,
        num_samples=train_size,
        length=16,
    )

    val_samples = (
        valid_size * specs.CLRS_30_ALGS_SETTINGS[algorithm]["num_samples_multiplier"]
    )

    # TODO: add if else statements for algos

    val_sampler, _ = clrs.build_sampler(
        name=algorithm,
        seed=valid_seed,
        num_samples=val_samples,
        length=16,
    )

    test_samples = (
        test_size * specs.CLRS_30_ALGS_SETTINGS[algorithm]["num_samples_multiplier"]
    )

    test_sampler, _ = clrs.build_sampler(
        name=algorithm,
        seed=test_seed,
        num_samples=test_samples,
        length=64,
    )

    return train_sampler, spec, val_sampler, test_sampler


def unpack(v):
    try:
        return v.item()  # DeviceArray
    except (AttributeError, ValueError):
        return v
