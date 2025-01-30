import * as React from "react";
import {
	AudioWaveform,
	GalleryVerticalEnd,
	SquareTerminal,
} from "lucide-react";

import { NavMain } from "~/components/nav-main";
import { RunSelector } from "~/components/run-selector";
import {
	Sidebar,
	SidebarContent,
	SidebarFooter,
	SidebarHeader,
	SidebarRail,
} from "~/components/ui/sidebar";
import path from "path";
import fs from "fs";
import {
	type LoaderFunctionArgs,
	useLoaderData,
	useSearchParams,
} from "react-router";

export const loader = async ({ request }: LoaderFunctionArgs) => {
	const directoryPath = path.join(process.cwd(), "./public/vids");
	const files = fs
		.readdirSync(directoryPath, { encoding: "utf-8" })
		.filter((file) => !file.includes(".DS_Store"));

	return {
		runs: files,
	};
};

export function AppSidebar({ ...props }: React.ComponentProps<typeof Sidebar>) {
	const { runs } = useLoaderData<typeof loader>();

	const [searchParams, setSearchParams] = useSearchParams();
	let run1 = searchParams.get("run1");
	let run2 = searchParams.get("run2");

	if (!run1 && runs.length > 0) {
		run1 = runs[0];
	}
	if (!run2 && runs.length > 1) {
		run2 = runs[1];
	}

	const navLinks = () => {
		return [
			{
				title: "Videos",
				url: "videos",
				icon: SquareTerminal,
				isActive: true,
				items: [
					{
						title: "Evaluation",
						url: `/videos/evaluation?run1=${run1}&run2=${run2}`,
					},
					{
						title: "Replay Buffer",
						url: `/videos/replay-buffer?run1=${run1}&run2=${run2}`,
					},
				],
			},
		];
	};

	return (
		<Sidebar collapsible="icon" {...props}>
			<SidebarHeader>
				<RunSelector
					index={1}
					runs={runs.map((r) => {
						return {
							name: r,
							logo: GalleryVerticalEnd,
						};
					})}
				/>
				<RunSelector
					index={2}
					runs={runs.map((r) => {
						return {
							name: r,
							logo: GalleryVerticalEnd,
						};
					})}
				/>
			</SidebarHeader>
			<SidebarContent>
				<NavMain items={navLinks()} />
			</SidebarContent>
			<SidebarFooter></SidebarFooter>
			<SidebarRail />
		</Sidebar>
	);
}
