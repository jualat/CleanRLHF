import React, { type ReactNode } from "react";

export const VideoGrid = ({ children }: { children: ReactNode }) => {
	return (
		<div className="container mx-auto py-8">
			<div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
				{children}
			</div>
		</div>
	);
};
