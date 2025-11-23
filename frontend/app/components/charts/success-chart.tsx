import { unifiedApi } from "@/app/types/api-types";
import { Card, CardBody, CircularProgress } from "@heroui/react";

const SuccessChart = ({ data }: { data: unifiedApi }) => {
  const { is_successful, success_probability_percentage } =
    data.rf_success_prob;
  const statusColor = is_successful ? "text-success-500" : "text-danger-500";

  return (
    <Card className="col-span-2">
      <CardBody className="flex flex-col items-center">
        <h3 className="self-start">Success Probability</h3>

        <CircularProgress
          classNames={{
            svg: "w-36 h-36 drop-shadow-md",
            indicator: "",
            track: "",
            value: "text-3xl font-semibold ",
          }}
          showValueLabel={true}
          strokeWidth={4}
          value={success_probability_percentage}
        />
        <p className="text-2xl">
          Business will{" "}
          <span className={statusColor}>
            {!is_successful && "not"} be successful
          </span>
        </p>
      </CardBody>
    </Card>
  );
};

export default SuccessChart;
