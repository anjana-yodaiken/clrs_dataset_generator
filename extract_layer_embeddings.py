from pathlib import Path
import jax
import jax.numpy as jnp
import clrs
from clrs._src import specs
import click

CWD = Path.cwd()
MODEL_PATH = Path(CWD, "trained_models/jarvis_march.pkl")


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
@click.option("--disable_edge_updates", default=False)
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
):
    main(
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


def unpack(v):
    try:
        return v.item()  # DeviceArray
    except (AttributeError, ValueError):
        return v


def evaluate(rng_key, model, feedback, spec, extras=None, verbose=False):
    """Evaluates a model on feedback."""
    out = {}
    predictions, aux = model.predict(rng_key, feedback.features)
    out.update(clrs.evaluate(feedback.outputs, predictions))
    if model.decode_hints and verbose:
        hint_preds = [clrs.decoders.postprocess(spec, x) for x in aux[0]]
        out.update(
            clrs.evaluate_hints(
                feedback.features.hints, feedback.features.lengths, hint_preds
            )
        )
    if extras:
        out.update(extras)
    if verbose:
        out.update(model.verbose_loss(feedback, aux))
    return {k: unpack(v) for k, v in out.items()}


def main(
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
    graph_vec="cat",
    processor_type="rt",
    checkpoint_path="tmp/CLRS30",
    freeze_processor=False,
):
    train_sampler, train_spec, val_sampler, test_sampler = get_dataset_samplers(
        algorithm, train_seed, valid_seed, test_seed, train_size, test_size, valid_size
    )
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
        hidden_dim=hidden_size,
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

    rt_model.init(feedback.features, model_seed + 1)
    rt_model.restore_model(str(MODEL_PATH), only_load_processor=False)
    rng_key = jax.random.PRNGKey(model_seed)
    test_sampler.reset_proc_samples()
    for sampler, batch_size in zip(
        [train_sampler, val_sampler, test_sampler],
        [batch_size, eval_batch_size, eval_batch_size],
    ):
        new_rng_key = get_model_embeddings(sampler, rt_model, batch_size, rng_key)
        rng_key = new_rng_key


def _iterate_sampler(sampler, batch_size):
    while True:
        yield sampler.next(batch_size)


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


if __name__ == "__main__":

    main()
