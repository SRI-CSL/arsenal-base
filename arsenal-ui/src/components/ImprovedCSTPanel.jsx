import React from 'react';
import JsonPanel from './JsonPanel';

export default class CSTPanel extends React.Component {

    render() {
        return <JsonPanel name="imp-cst-panel" title="Final CSTs" data={this.props.cst}/>;
    }

}
