import * as React from "react";
import { ChevronsUpDown, type LucideIcon } from "lucide-react";

import {
	DropdownMenu,
	DropdownMenuContent,
	DropdownMenuItem,
	DropdownMenuTrigger,
} from "~/components/ui/dropdown-menu";
import {
	SidebarMenu,
	SidebarMenuButton,
	SidebarMenuItem,
	useSidebar,
} from "~/components/ui/sidebar";
import { useState } from "react";
import { useSearchParams } from "react-router";

interface Run {
	name: string;
	logo: LucideIcon;
}

interface RunSelectorProps {
	runs: Run[];
	index: number;
}

export function RunSelector({ runs, index }: RunSelectorProps) {
	const [searchParams, setSearchParams] = useSearchParams();
	const run = searchParams.get(`run${index}`);
	const [activeRun, setActiveRun] = useState(runs.find((r) => r.name === run));

	return (
		<SidebarMenu>
			<SidebarMenuItem>
				<DropdownMenu modal={false}>
					<DropdownMenuTrigger asChild>
						{activeRun && (
							<SidebarMenuButton
								size="lg"
								className="data-[state=open]:bg-sidebar-accent data-[state=open]:text-sidebar-accent-foreground"
							>
								<div className="flex aspect-square size-8 items-center justify-center rounded-lg bg-sidebar-primary text-sidebar-primary-foreground">
									<activeRun.logo className="size-4" />
								</div>
								<div className="grid flex-1 text-left text-sm leading-tight">
									<span className="truncate font-semibold">
										{activeRun.name}
									</span>
								</div>
								<ChevronsUpDown className="ml-auto" />
							</SidebarMenuButton>
						)}
					</DropdownMenuTrigger>
					<DropdownMenuContent className="w-[--radix-popper-anchor-width] bg-white overflow-y-scroll h-96">
						{runs.map((run) => (
							<DropdownMenuItem
								key={run.name}
								onClick={() => {
									setActiveRun(run);
									searchParams.set(`run${index}`, run.name);
									setSearchParams(searchParams);
								}}
							>
								<span>{run.name}</span>
							</DropdownMenuItem>
						))}
					</DropdownMenuContent>
				</DropdownMenu>
			</SidebarMenuItem>
		</SidebarMenu>
	);
}
