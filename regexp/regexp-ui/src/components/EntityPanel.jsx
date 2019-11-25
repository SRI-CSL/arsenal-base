import React from 'react';
import JsonPanel from './JsonPanel';

export default class IRPanel extends React.Component {

    render() {
        return <JsonPanel name="entity-panel" title="Entities" data={this.props.entities}/>;
    }

}
