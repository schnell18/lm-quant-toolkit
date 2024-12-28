#!/usr/bin/env python3

import re
from argparse import ArgumentParser
from datetime import datetime
from timeit import default_timer as timer

import pandas as pd
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_community.llms import VLLMOpenAI
from langchain_core.output_parsers.list import CommaSeparatedListOutputParser


def generate_level1_topic(model, variant, topic, count=10):
    output_parser = CommaSeparatedListOutputParser()
    format_instructions = output_parser.get_format_instructions()
    template = """
Please divide the {topic} into {count} most important sub-topics.
Number each sub-topic. Give answers without any extra description
or explanation. {format_instructions}
"""
    prompt = PromptTemplate(
        template=template,
        input_variables=["topic", "count"],
        partial_variables={"format_instructions": format_instructions},
    )

    llm = _get_llm_model(model, variant)

    chain = prompt | llm | output_parser
    result = chain.invoke({"count": count, "topic": topic})
    return result


def generate_level2_topic(model, variant, topic):
    template = """
Please explain {topic} concisely in around 100 to 120 words.
Give the answer directly without any extra words such as
"sure", "certainly", "of course" etc. Don't begin the answer with
words such as "system", "expert" etc to indicate role.
"""
    prompt = PromptTemplate(
        template=template,
        input_variables=["topic"],
    )

    llm = _get_llm_model(model, variant)

    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"topic": topic})

    return result


def parse_level1_output(topic, output):
    pat = re.compile(r"^\d+\.\s+")
    ret = []
    for topic in output:
        m = re.match(pat, topic)
        if m:
            ret.append(topic[m.end(0) :])
    return ret


def parse_output(model, topic, text):
    instructions = text.split("\n")
    dict_list = []
    for inst in instructions:
        if len(inst) < 60:
            continue
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_task = {
            "topic": topic,
            "timestamp": ts,
            "text": inst,
        }
        dict_list.append(new_task)
    return dict_list


def _get_llm_model(model, variant):
    return VLLMOpenAI(
        model_name=model,
        openai_api_key="token-abc123",
        openai_api_base="http://localhost:8000/v1",
        temperature=0.6,
    )


def save_llm_raw_output(llm_output_dir, model, variant, topic, text):
    short_id = model.split("/")[1]
    topic_str = topic.replace(" ", "-").replace("/", "-")
    file_name = f"{llm_output_dir}/output-{topic_str}-{short_id}-{variant}.log"
    with open(file_name, "w") as writer:
        writer.write(text)


def generate(plans, llm_output_dir, model="llama", variant="7b", trace=True):
    t0 = timer()
    dict_list = []
    for plan in plans:
        ti0 = timer()
        topic = plan["topic"]
        output = generate_level1_topic(model, variant, topic, count=plan["count"])
        ti1 = timer()
        sub_topics = parse_level1_output(topic, output)
        for sub_topic in sub_topics:
            output = generate_level2_topic(model, variant, sub_topic)
            if trace:
                save_llm_raw_output(
                    llm_output_dir,
                    model,
                    variant,
                    sub_topic,
                    output,
                )
                print(f"{sub_topic} generation took {(ti1 - ti0):.2f} seconds!")
            records = parse_output(model, sub_topic, output)
            dict_list.extend(records)
    df = pd.DataFrame(dict_list)
    short_id = model.split("/")[1]
    file_name = f"data-{short_id}-{variant}.csv"
    df.to_csv(
        file_name, columns=["topic", "text", "timestamp"], index=False, doublequote=True
    )

    t1 = timer()
    if trace:
        print(f"All instructions generation took {t1 - t0} seconds")


def _parse_args():
    # Create the parser
    parser = ArgumentParser(description="Kubernetes SFT Dataset Generator")

    # Add arguments
    parser.add_argument(
        "-m",
        "--model",
        required=True,
        help="LLM model name, eg: claude, gpt, mistral, llama2",
    )
    parser.add_argument(
        "-v", "--variant", required=True, help="model variant such as opus, sonnet"
    )
    parser.add_argument(
        "-d", "--trace", action="store_true", default=False, help="Print trace messages"
    )
    parser.add_argument(
        "--topic-file",
        default="topics.txt",
        help="The Kubernetes topics for generated instructions",
    )
    parser.add_argument(
        "--llm-output-dir",
        default="results",
        help="The directory to save raw LLM output",
    )

    # Parse the arguments
    args = parser.parse_args()
    return args


def _load_topics(file):
    plans = []
    with open(file, "r") as f:
        for line in f.readlines():
            topic, count = line.split(":")
            topic = topic.replace('"', "")
            plans.append(
                {
                    "topic": topic,
                    "count": int(count.rstrip()),
                }
            )
    return plans


if __name__ == "__main__":
    args = _parse_args()
    print(args)

    model = args.model
    variant = args.variant
    trace = args.trace
    llm_output_dir = args.llm_output_dir
    plans = _load_topics(args.topic_file)

    generate(plans, llm_output_dir, model, variant, trace)
