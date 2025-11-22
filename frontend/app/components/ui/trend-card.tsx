import { Card, Chip } from "@heroui/react";
import { MoveRightIcon, TrendingUpIcon } from "lucide-react";

type TrendCardProps = {
  title: string;
  value: string;
  change: string;
  changeType: "positive" | "neutral" | "negative";
  trendChipVariant?: "flat" | "light";
};

const TrendCard = ({
  title,
  value,
  change,
  changeType,
  trendChipVariant = "light",
}: TrendCardProps) => {
  return (
    <Card className="dark:border-default-100 border border-transparent">
      <div className="flex p-4">
        <div className="flex flex-col gap-y-2">
          <dt className="text-small text-default-500 font-medium">{title}</dt>
          <dd className="text-default-700 text-2xl font-semibold">{value}</dd>
        </div>
        <Chip
          className={"absolute right-4 top"}
          classNames={{
            content: "font-medium text-[0.65rem]",
          }}
          color={
            changeType === "positive"
              ? "success"
              : changeType === "neutral"
              ? "warning"
              : "danger"
          }
          radius="sm"
          size="sm"
          variant={trendChipVariant}
        >
          {change}
        </Chip>
      </div>
    </Card>
  );
};

export default TrendCard;
