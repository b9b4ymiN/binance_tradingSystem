import { ReactNode } from "react";
import { Card, CardContent } from "./Card";
import {
  cn,
  formatCurrency,
  formatPercent,
  getPercentageColor,
  getPercentageIcon,
} from "../lib/utils";

interface MetricCardProps {
  title: string;
  value: string | number;
  change?: number;
  changeType?: "currency" | "percentage" | "number";
  icon?: ReactNode;
  color?: "default" | "success" | "danger" | "warning";
  loading?: boolean;
}

export function MetricCard({
  title,
  value,
  change,
  changeType = "percentage",
  icon,
  color = "default",
  loading = false,
}: MetricCardProps) {
  const colorClasses = {
    default: "border-gray-200",
    success: "border-success-200 bg-success-50/50",
    danger: "border-danger-200 bg-danger-50/50",
    warning: "border-warning-200 bg-warning-50/50",
  };

  const formatChange = (val: number) => {
    switch (changeType) {
      case "currency":
        return formatCurrency(val);
      case "percentage":
        return formatPercent(val);
      case "number":
        return (val).toString();
      default:
        return val.toString();
    }
  };

  if (loading) {
    return (
      <Card className={cn("animate-pulse", colorClasses[color])}>
        <CardContent>
          <div className="flex items-center justify-between">
            <div className="space-y-2">
              <div className="h-4 bg-gray-200 rounded w-20"></div>
              <div className="h-8 bg-gray-200 rounded w-24"></div>
            </div>
            {icon && <div className="h-8 w-8 bg-gray-200 rounded"></div>}
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card
      className={cn("transition-all duration-200", colorClasses[color])}
      hover
    >
      <CardContent>
        <div className="flex items-center justify-between">
          <div className="space-y-1">
            <p className="text-sm font-medium text-gray-600">{title}</p>
            <p className="text-2xl font-bold text-gray-900">
              {typeof value === "number" ? (value) : value}
            </p>
            {change !== undefined && (
              <div
                className={cn(
                  "flex items-center text-sm",
                  getPercentageColor(change)
                )}
              >
                <span className="mr-1">{getPercentageIcon(change)}</span>
                <span>{formatChange(Math.abs(change))}</span>
                <span className="ml-1 text-gray-500">today</span>
              </div>
            )}
          </div>
          {icon && <div className="h-8 w-8 text-gray-400">{icon}</div>}
        </div>
      </CardContent>
    </Card>
  );
}
