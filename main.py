import jax
import jax.numpy as jnp
import clrs
import click
from utils import _iterate_sampler, restore_model, get_dataset_samplers, unpack


@click.command()
@click.option("--algorithm", default="jarvis_march")
@click.option("--train_seed", default=1)
@click.option("--valid_seed", default=2)
@click.option("--test_seed", default=3)
@click.option("--model_seed", default=42)
@click.option("--batch_size", default=4)
@click.option("--eval_batch_size", default=1)
@click.option("--chunked_training", default=False)
@click.option("--chunk_length", default=100)
@click.option("--train_items", default=320000)
@click.option("--train_size", default=10000)
@click.option("--valid_size", default=32)
@click.option("--test_size", default=32)
@click.option("--eval_every", default=50)
@click.option("--eval_on_train_set", default=True)
@click.option("--eval_on_test_set", default=True)
@click.option("--verbose_logging", default=False)
@click.option("--log_param_count", default=False)
@click.option("--ptr_from_edges", default=True)
@click.option("--disable_edge_updates", default=True)
@click.option("--num_layers", default=3)
@click.option("--hidden_size", default=0)
@click.option("--learning_rate", default=0.00025)
@click.option("--dropout_prob", default=0.0)
@click.option("--hint_teacher_forcing_noise", default=0.5)
@click.option("--nb_heads", default=12)
@click.option("--head_size", default=16)
@click.option("--node_hid_size", default=32)
@click.option("--edge_hid_size_1", default=16)
@click.option("--edge_hid_size_2", default=8)
@click.option("--hint_mode", default="encoded_decoded_nodiff")
@click.option("--use_ln", default=True)
@click.option("--use_lstm", default=False)
@click.option("--graph_vec", default="cat")
@click.option("--processor_type", default="rt")
@click.option("--checkpoint_path", default="/tmp/CLRS30")
@click.option("--freeze_processor", default=False)
@click.option("--function", default="extract")
@click.option("--model_path", default=None)
def click_main(
    algorithm,
    train_seed,
    valid_seed,
    test_seed,
    model_seed,
    batch_size,
    eval_batch_size,
    chunked_training,
    chunk_length,
    train_items,
    train_size,
    test_size,
    valid_size,
    eval_every,
    eval_on_train_set,
    eval_on_test_set,
    verbose_logging,
    log_param_count,
    ptr_from_edges,
    disable_edge_updates,
    num_layers,
    hidden_size,
    learning_rate,
    dropout_prob,
    hint_teacher_forcing_noise,
    nb_heads,
    head_size,
    node_hid_size,
    edge_hid_size_1,
    edge_hid_size_2,
    hint_mode,
    use_ln,
    use_lstm,
    graph_vec,
    processor_type,
    checkpoint_path,
    freeze_processor,
    function,
    model_path,
):
    if function == "extract":
        main_extract_layer_embeddings(
            algorithm,
            train_seed,
            valid_seed,
            test_seed,
            model_seed,
            batch_size,
            eval_batch_size,
            chunked_training,
            chunk_length,
            train_items,
            train_size,
            test_size,
            valid_size,
            eval_every,
            eval_on_train_set,
            eval_on_test_set,
            verbose_logging,
            log_param_count,
            ptr_from_edges,
            disable_edge_updates,
            num_layers,
            hidden_size,
            learning_rate,
            dropout_prob,
            hint_teacher_forcing_noise,
            nb_heads,
            head_size,
            node_hid_size,
            edge_hid_size_1,
            edge_hid_size_2,
            hint_mode,
            use_ln,
            use_lstm,
            graph_vec,
            processor_type,
            checkpoint_path,
            freeze_processor,
        )

    if function == "validate":
        main_validate_model(
            algorithm,
            train_seed,
            valid_seed,
            test_seed,
            model_seed,
            batch_size,
            eval_batch_size,
            chunked_training,
            chunk_length,
            train_items,
            train_size,
            test_size,
            valid_size,
            eval_every,
            eval_on_train_set,
            eval_on_test_set,
            verbose_logging,
            log_param_count,
            ptr_from_edges,
            disable_edge_updates,
            num_layers,
            hidden_size,
            learning_rate,
            dropout_prob,
            hint_teacher_forcing_noise,
            nb_heads,
            head_size,
            node_hid_size,
            edge_hid_size_1,
            edge_hid_size_2,
            hint_mode,
            use_ln,
            use_lstm,
            graph_vec,
            processor_type,
            checkpoint_path,
            freeze_processor,
            model_path,
        )


def evaluate_preds(preds, outputs, hints, lengths, hint_preds, spec, extras):
    """Evaluates predictions against feedback."""
    out = {}
    out.update(clrs.evaluate(outputs, preds))
    if hint_preds:
        hint_preds = [clrs.decoders.postprocess(spec, x) for x in hint_preds]
        out.update(clrs.evaluate_hints(hints, lengths, hint_preds))
    if extras:
        out.update(extras)
    return {k: unpack(v) for k, v in out.items()}


def _concat(dps, axis):
    return jax.tree_util.tree_map(lambda *x: jnp.concatenate(x, axis), *dps)


def collect_and_eval(
    sampler,
    predict_fn,
    sample_count,
    rng_key,
    spec,
    extras,
    batch_size,
    verbose_logging,
):
    """Collect batches of output and hint preds and evaluate them."""
    verbose = verbose_logging
    processed_samples = 0
    preds = []
    hint_preds = []
    outputs = []
    hints = []
    lengths = []
    while processed_samples < sample_count:
        try:
            feedback = next(sampler)
        except TypeError as te:
            feedback = sampler.next(batch_size, eval=True)
        outputs.append(feedback.outputs)
        rng_key, new_rng_key = jax.random.split(rng_key)
        cur_preds, (cur_hint_preds, _, _) = predict_fn(rng_key, feedback.features)
        preds.append(cur_preds)
        if verbose:
            hints.append(feedback.features.hints)
            lengths.append(feedback.features.lengths)
            hint_preds.append(cur_hint_preds)
        rng_key = new_rng_key
        processed_samples += batch_size
    sampler.reset_proc_samples()
    outputs = _concat(outputs, axis=0)
    preds = _concat(preds, axis=0)
    if verbose:
        # for hints, axis=1 because hints have time dimension first
        hints = _concat(hints, axis=1)
        lengths = _concat(lengths, axis=0)
        # for hint_preds, axis=0 because the time dim is unrolled as a list
        hint_preds = _concat(hint_preds, axis=0)

    return evaluate_preds(preds, outputs, hints, lengths, hint_preds, spec, extras)


def main_validate_model(
    algorithm="jarvis_march",
    train_seed=1,
    valid_seed=2,
    test_seed=3,
    model_seed=42,
    batch_size=4,
    eval_batch_size=1,
    chunked_training=False,
    chunk_length=100,
    train_items=320000,
    train_size=10000,
    test_size=32,
    valid_size=32,
    eval_every=50,
    eval_on_train_set=True,
    eval_on_test_set=True,
    verbose_logging=False,
    log_param_count=True,
    ptr_from_edges=True,
    disable_edge_updates=False,
    num_layers=3,
    hidden_size=0,
    learning_rate=0.00025,
    dropout_prob=0.0,
    hint_teacher_forcing_noise=0.5,
    nb_heads=12,
    head_size=16,
    node_hid_size=32,
    edge_hid_size_1=16,
    edge_hid_size_2=8,
    hint_mode="encoded_decoded_nodiff",
    use_ln=True,
    use_lstm=False,
    graph_vec="att",
    processor_type="rt",
    checkpoint_path="tmp/CLRS30",
    freeze_processor=False,
    encoder_decoder_path=None,
    model_path=None,
):
    train_sampler, train_spec, val_sampler, test_sampler = get_dataset_samplers(
        algorithm, train_seed, valid_seed, test_seed, train_size, test_size, valid_size
    )

    dataset_specs = {
        "train": {
            "sampler": train_sampler,
            "batch_size": batch_size,
            "n_samples": train_sampler._num_samples,
            "save_emb_sub_dir": "",  # not saving embeddings
        },
        "val": {
            "sampler": val_sampler,
            "batch_size": eval_batch_size,
            "n_samples": val_sampler._num_samples,
            "save_emb_sub_dir": "",  # not saving embeddings
        },
        "test": {
            "sampler": test_sampler,
            "batch_size": eval_batch_size,
            "n_samples": test_sampler._num_samples,
            "save_emb_sub_dir": "",  # not saving embeddings
        },
    }
    results = {}
    for name, specs in dataset_specs.items():
        sampler = specs["sampler"]
        n_samples = specs["n_samples"]
        batch_size = specs["batch_size"]
        save_emb_sub_dir = specs["save_emb_sub_dir"]

        rng_key = jax.random.PRNGKey(model_seed)
        rt_model = restore_model(
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
            False,  # save_embeddings
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
            encoder_decoder_path=encoder_decoder_path,
            model_path=model_path,
        )

        stats = collect_and_eval(
            sampler,
            rt_model.predict,
            n_samples,
            rng_key,
            batch_size=batch_size,
            verbose_logging=verbose_logging,
            spec=train_spec,
            extras={},
        )
        print(processor_type)
        print(f"{name}: ", stats)
        results[name] = stats


def main_extract_layer_embeddings(
    algorithm="jarvis_march",
    train_seed=1,
    valid_seed=2,
    test_seed=3,
    model_seed=42,
    batch_size=4,
    eval_batch_size=1,
    chunked_training=False,
    chunk_length=100,
    train_items=320000,
    train_size=10000,
    test_size=32,
    valid_size=32,
    eval_every=50,
    eval_on_train_set=True,
    eval_on_test_set=True,
    verbose_logging=False,
    log_param_count=True,
    ptr_from_edges=True,
    disable_edge_updates=False,
    num_layers=3,
    hidden_size=0,
    learning_rate=0.00025,
    dropout_prob=0.0,
    hint_teacher_forcing_noise=0.5,
    nb_heads=12,
    head_size=16,
    node_hid_size=32,
    edge_hid_size_1=16,
    edge_hid_size_2=8,
    hint_mode="encoded_decoded_nodiff",
    use_ln=True,
    use_lstm=False,
    graph_vec="att",
    processor_type="rt",
    checkpoint_path="tmp/CLRS30",
    freeze_processor=False,
    model_path="trained_models/rt_jarvis_march.pkl",
):
    train_sampler, train_spec, val_sampler, test_sampler = get_dataset_samplers(
        algorithm, train_seed, valid_seed, test_seed, train_size, test_size, valid_size
    )

    dataset_specs = {
        "train": {
            "sampler": train_sampler,
            "batch_size": batch_size,
            "save_emb_sub_dir": "train",
        },
        "val": {
            "sampler": val_sampler,
            "batch_size": eval_batch_size,
            "save_emb_sub_dir": "val",
        },
        "test": {
            "sampler": test_sampler,
            "batch_size": eval_batch_size,
            "save_emb_sub_dir": "test",
        },
    }
    for name, specs in dataset_specs.items():
        sampler = specs["sampler"]
        batch_size = specs["batch_size"]
        save_emb_sub_dir = specs["save_emb_sub_dir"]

        rng_key = jax.random.PRNGKey(model_seed)

        rt_model = restore_model(
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
            True,  # save_embeddings
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
            encoder_decoder_path=None,
            model_path=model_path,
        )

        new_rng_key = get_model_embeddings(sampler, rt_model, batch_size, rng_key)
        rng_key = new_rng_key


def get_model_embeddings(sampler, model, batch_size, rng_key):
    counter = 0
    for _ in range(0, sampler._num_samples, batch_size):

        feedback_gen = _iterate_sampler(sampler, batch_size)
        feedback = next(feedback_gen)

        rng_key, new_rng_key = jax.random.split(rng_key)
        _, _ = model.predict(rng_key, feedback.features)
        counter = counter + batch_size
        rng_key = new_rng_key

    return rng_key


if __name__ == "__main__":
    main_extract_layer_embeddings()
