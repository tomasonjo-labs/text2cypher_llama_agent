from workflows.iterative_planner import IterativePlanningFlow
from workflows.naive_text2cypher import NaiveText2CypherFlow
from workflows.naive_text2cypher_retry import NaiveText2CypherRetryFlow
from workflows.text2cypher_retry_check import NaiveText2CypherRetryCheckFlow
from workflows.bizdev import BizDev

WORKFLOW_MAP = {
    "text2cypher_with_1_retry_and_output_check": NaiveText2CypherRetryCheckFlow,
    "naive_text2cypher": NaiveText2CypherFlow,
    "naive_text2cypher_with_1_retry": NaiveText2CypherRetryFlow,
    "chat+viz": BizDev
    # "iterative_planning": IterativePlanningFlow,
}
